"""Provisional static admission followed by isolated background workflow checks."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from model_preflight.core.enums import OverallStatus, StageStatus
from model_preflight.core.exceptions import ManifestError, ProfileError
from model_preflight.core.results import PreflightReport, StageResult
from model_preflight.engine.cache import cache_key
from model_preflight.engine.runner import PreflightRunner, _hash_tree
from model_preflight.manifest.loader import load_manifest


@dataclass(frozen=True)
class CheckUpdate:
    """Versioned integration envelope; STATIC_PASS is never a final report."""

    request_id: str
    candidate_id: str
    candidate_hash: str
    fixture_hash: str
    manifest_hash: str
    status: str
    sequence: int
    static_result: StageResult
    report: PreflightReport | None = None
    error: dict[str, str] | None = None

    @property
    def terminal(self) -> bool:
        return self.status != "STATIC_PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_schema_version": 1,
            "request_id": self.request_id,
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "fixture_hash": self.fixture_hash,
            "manifest_hash": self.manifest_hash,
            "sequence": self.sequence,
            "status": self.status,
            "terminal": self.terminal,
            "speculative_submission_allowed": not self.terminal,
            "gpu_check_required": self.report.gpu_check_required if self.report else True,
            "static_result": self.static_result.to_dict(),
            "report": self.report.to_dict() if self.report else None,
            "error": self.error,
        }


@dataclass(frozen=True)
class PreflightSubmission:
    """Local handle. Keep the owning process alive until completion resolves."""

    initial: CheckUpdate
    completion: Future[CheckUpdate]

    def poll(self) -> CheckUpdate:
        return self.completion.result() if self.completion.done() else self.initial


def submit(manifest_path: str | Path) -> PreflightSubmission:
    """Return after static checks and run the complete workflow in the background.

    Only trusted local inputs should use this API directly. For generated code,
    consume ``check --format events`` from a restricted CPU container instead.
    Inputs must remain immutable; speculative runs always bypass the report cache.
    """
    path = Path(manifest_path).resolve()
    try:
        manifest_source = path.read_bytes()
    except OSError as exc:
        raise ManifestError(str(exc)) from exc
    manifest = load_manifest(path)
    try:
        runner = PreflightRunner(manifest, use_cache=False)
    except ProfileError as exc:
        raise ManifestError(str(exc)) from exc
    static = runner.run_static()
    initial = CheckUpdate(
        request_id=uuid4().hex,
        candidate_id=manifest.candidate.id,
        candidate_hash=runner.candidate_hash,
        fixture_hash=runner.fixture_hash,
        manifest_hash=cache_key(manifest.to_dict()),
        status="STATIC_PASS",
        sequence=1,
        static_result=deepcopy(static),
    )

    def inputs_unchanged() -> bool:
        try:
            return (
                path.read_bytes() == manifest_source
                and _hash_tree(manifest.candidate.root) == runner.candidate_hash
                and _hash_tree(manifest.task.fixture_root) == runner.fixture_hash
            )
        except OSError:
            return False

    def finish(sequence: int) -> CheckUpdate:
        try:
            if not inputs_unchanged():
                return changed(sequence)
            report = runner.run(static_result=static)
            if not inputs_unchanged():
                return changed(sequence)
            return replace(
                initial,
                sequence=sequence,
                status="FULL_PASS" if report.overall_status is OverallStatus.PASS
                else report.overall_status.value,
                report=report,
            )
        except Exception as exc:
            return replace(
                initial, sequence=sequence, status="INTERNAL_ERROR",
                error={"code": "CONTROLLER_ERROR", "message": f"{type(exc).__name__}: {exc}"},
            )

    def changed(sequence: int) -> CheckUpdate:
        return replace(
            initial, sequence=sequence, status="INCONCLUSIVE",
            error={
                "code": "INPUT_CHANGED",
                "message": "Candidate, fixture, or manifest changed; resubmit immutable inputs.",
            },
        )

    if static.status is not StageStatus.PASS:
        # A failed static gate never imports the candidate or starts a worker.
        runner.only = {"static_source"}
        terminal = finish(1)
        completion: Future[CheckUpdate] = Future()
        completion.set_result(terminal)
        return PreflightSubmission(terminal, completion)

    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preflight")
    try:
        completion = pool.submit(finish, 2)
    finally:
        # Shutdown releases the pool after this task; it must not block submit().
        pool.shutdown(wait=False)
    return PreflightSubmission(initial, completion)
