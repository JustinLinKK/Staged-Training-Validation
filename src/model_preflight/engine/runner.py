"""Controller orchestration. Candidate code is never imported here."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from model_preflight.checks.hardware import load_profile, run_hardware_checks
from model_preflight.checks.numeric import estimate_resources
from model_preflight.checks.source import run_static_checks
from model_preflight.core.enums import Classification, OverallStatus, StageName, StageStatus
from model_preflight.core.policies import status_from_diagnostics, submission_recommendation
from model_preflight.core.results import PreflightReport, StageResult
from model_preflight.engine.cache import ResultCache, cache_key
from model_preflight.engine.registry import STAGE_ORDER, validate_stage_names
from model_preflight.engine.stage import combine_stage_results
from model_preflight.execution.subprocess_runner import SubprocessRunner
from model_preflight.execution.worker_protocol import SHAPE_REDUCER_REVISION, WorkerRequest
from model_preflight.manifest.models import Manifest
from model_preflight.version import __version__


def _hash_tree(root: Path | None) -> str:
    digest = hashlib.sha256()
    if root is None:
        return f"sha256:{digest.hexdigest()}"
    for path in sorted(root.rglob("*")):
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if any(
            part in {".git", "__pycache__", ".model-preflight-cache"}
            for part in relative.parts
        ):
            continue
        if not path.is_file() and not path.is_symlink():
            continue
        digest.update(str(relative).encode("utf-8"))
        if path.is_symlink():
            digest.update(b"<symlink>")
            digest.update(os.fsencode(path.readlink()))
        if not path.is_file():
            continue
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<unreadable>")
    return f"sha256:{digest.hexdigest()}"


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _timed(function: Any, *args: Any) -> StageResult:
    started = time.monotonic()
    result: StageResult = function(*args)
    result.duration_seconds = time.monotonic() - started
    return result


def _skipped(name: str, reason: str) -> StageResult:
    return StageResult(
        name=name,
        status=StageStatus.SKIPPED,
        evidence={"reason": reason},
    )


class PreflightRunner:
    def __init__(
        self,
        manifest: Manifest,
        *,
        only: set[str] | None = None,
        use_cache: bool | None = None,
    ) -> None:
        self.manifest = manifest
        self.only = set(STAGE_ORDER) if only is None else set(only)
        validate_stage_names(self.only)
        self.use_cache = manifest.execution.cache if use_cache is None else use_cache
        self.profile = load_profile(manifest)
        self.candidate_hash = _hash_tree(manifest.candidate.root)
        self.fixture_hash = _hash_tree(manifest.task.fixture_root)
        self.subprocess = SubprocessRunner(manifest.execution)

    def _worker(
        self, stage: StageName, scenario: dict[str, Any], timeout: float,
        *, reduction_allowed: bool = False,
    ) -> StageResult:
        result = self.subprocess.run(
            WorkerRequest(
                stage=stage.value,
                manifest=self.manifest.to_dict(),
                scenario=scenario,
                shape_reduction_allowed=reduction_allowed,
            ),
            timeout,
        )
        reduction = result.evidence.get("shape_reduction")
        if (
            reduction_allowed and result.status is not StageStatus.PASS
            and (reduction is None or reduction.get("applied", False))
        ):
            original = self._worker(stage, scenario, timeout)
            original.evidence["reduction_attempt"] = result.to_dict()
            original.evidence["shape_reduction"] = {
                **original.evidence.get("shape_reduction", {}),
                "applied": False,
                "execution_model": "original",
                "reason": "original_width_retry",
            }
            original.duration_seconds += result.duration_seconds
            return original
        return result

    def _scenarios(
        self, stage: StageName, scenarios: list[dict[str, Any]], timeout: float,
        *, reduction_allowed: bool = False,
    ) -> StageResult:
        def run(scenario: dict[str, Any]) -> StageResult:
            return self._worker(stage, scenario, timeout, reduction_allowed=reduction_allowed)

        workers = min(self.manifest.execution.max_workers, len(scenarios))
        started = time.monotonic()
        if workers <= 1:
            results = [run(scenario) for scenario in scenarios]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                # map preserves manifest order regardless of completion order.
                results = list(pool.map(run, scenarios))
        result = combine_stage_results(stage.value, results)
        if workers > 1:
            result.evidence["worker_seconds"] = result.duration_seconds
            result.duration_seconds = time.monotonic() - started
            result.evidence["parallel_workers"] = workers
        return result

    def _cache_inputs(self) -> dict[str, Any]:
        return {
            "candidate_hash": self.candidate_hash,
            "fixture_hash": self.fixture_hash,
            "manifest": self.manifest.to_dict(),
            "checker_version": __version__,
            "torch_version": _distribution_version("torch"),
            "profile": self.profile,
            "stages": sorted(self.only),
            "shape_reducer_revision": SHAPE_REDUCER_REVISION,
        }

    def run_static(self) -> StageResult:
        return _timed(run_static_checks, self.manifest)

    def run(self, *, static_result: StageResult | None = None) -> PreflightReport:
        key = cache_key(self._cache_inputs())
        cache = ResultCache()
        if self.use_cache:
            cached = cache.load(key)
            if cached is not None:
                return cached

        stages: list[StageResult] = []
        by_name: dict[str, StageResult] = {}
        normal_scenario = self.manifest.scenarios.cpu_scenarios()[0]

        if StageName.STATIC_SOURCE.value in self.only:
            result = self.run_static() if static_result is None else static_result
            stages.append(result)
            by_name[result.name] = result
        static_failed = (
            by_name.get(StageName.STATIC_SOURCE.value, _skipped("", "")).status
            is StageStatus.FAIL
        )

        if StageName.HARDWARE.value in self.only:
            result = _timed(run_hardware_checks, self.manifest, self.profile)
            stages.append(result)
            by_name[result.name] = result

        runtime_names = {
            StageName.CONSTRUCTION.value,
            StageName.DATA_CONTRACT.value,
            StageName.ABSTRACT_FORWARD.value,
            StageName.CPU_TRAINING.value,
            StageName.VALIDATION.value,
        }
        if static_failed:
            for name in STAGE_ORDER:
                if name in self.only and name in runtime_names:
                    result = _skipped(name, "static source failure prevents safe import")
                    stages.append(result)
                    by_name[name] = result
        else:
            if StageName.CONSTRUCTION.value in self.only:
                result = self._worker(
                    StageName.CONSTRUCTION,
                    normal_scenario,
                    self.manifest.execution.cpu_timeout_seconds,
                )
                stages.append(result)
                by_name[result.name] = result

            if StageName.DATA_CONTRACT.value in self.only:
                result = self._worker(
                    StageName.DATA_CONTRACT,
                    normal_scenario,
                    self.manifest.execution.cpu_timeout_seconds,
                )
                stages.append(result)
                by_name[result.name] = result

            if StageName.ABSTRACT_FORWARD.value in self.only:
                result = self._scenarios(
                    StageName.ABSTRACT_FORWARD,
                    self.manifest.scenarios.matrix(),
                    self.manifest.execution.abstract_timeout_seconds,
                )
                stages.append(result)
                by_name[result.name] = result

            reduction_allowed = (
                self.manifest.execution.shape_reduction.mode == "auto"
                and StageName.STATIC_SOURCE.value in by_name
                and all(
                    name in by_name and by_name[name].status is StageStatus.PASS
                    for name in ("construction", "data_contract", "abstract_forward")
                )
            )
            if StageName.CPU_TRAINING.value in self.only:
                cpu_scenarios = self.manifest.scenarios.cpu_scenarios()
                if self.manifest.task.drop_last:
                    cpu_scenarios = cpu_scenarios[:1]
                result = self._scenarios(
                    StageName.CPU_TRAINING,
                    cpu_scenarios,
                    self.manifest.execution.cpu_timeout_seconds,
                    reduction_allowed=reduction_allowed,
                )
                stages.append(result)
                by_name[result.name] = result

            if StageName.VALIDATION.value in self.only:
                if self.manifest.scenarios.run_validation:
                    validation_scenario = {**normal_scenario, "mode": "validation"}
                    result = self._worker(
                        StageName.VALIDATION,
                        validation_scenario,
                        self.manifest.execution.cpu_timeout_seconds,
                        reduction_allowed=reduction_allowed,
                    )
                else:
                    result = _skipped(
                        StageName.VALIDATION.value,
                        "scenarios.run_validation is false",
                    )
                stages.append(result)
                by_name[result.name] = result

        resource_estimate: dict[str, Any] | None = None
        if StageName.MEMORY.value in self.only:
            construction = by_name.get(StageName.CONSTRUCTION.value)
            if construction is None and not static_failed:
                construction = self._worker(
                    StageName.CONSTRUCTION,
                    normal_scenario,
                    self.manifest.execution.cpu_timeout_seconds,
                )
            if construction is None or construction.status is not StageStatus.PASS:
                result = _skipped(
                    StageName.MEMORY.value,
                    "memory estimate requires successful construction",
                )
            else:
                activation_bytes = 0
                abstract = by_name.get(StageName.ABSTRACT_FORWARD.value)
                if abstract is not None:
                    activation_bytes = max(
                        (
                            int(item.get("estimated_forward_activation_bytes", 0))
                            for item in abstract.evidence.get("scenarios", [])
                        ),
                        default=0,
                    )
                result, resource_estimate = estimate_resources(
                    self.manifest,
                    self.profile,
                    construction.evidence,
                    activation_bytes,
                )
            stages.append(result)
            by_name[result.name] = result

        diagnostics = [diagnostic for stage in stages for diagnostic in stage.diagnostics]
        status = status_from_diagnostics(diagnostics)
        gpu_check_required = any(
            item.classification is Classification.INCONCLUSIVE for item in diagnostics
        )
        gpu_check_required = gpu_check_required or any(
            evidence.get("shape_reduction", {}).get("applied", False)
            for stage in stages
            for evidence in stage.evidence.get("scenarios", [stage.evidence])
        )
        recommendation = submission_recommendation(status, self.manifest.policy.mode)
        memory = by_name.get(StageName.MEMORY.value)
        if memory is not None and memory.status is StageStatus.INCONCLUSIVE:
            recommendation = self.manifest.policy.mode.value == "audit"
        report = PreflightReport(
            candidate_id=self.manifest.candidate.id,
            candidate_hash=self.candidate_hash,
            checker_version=__version__,
            overall_status=status,
            gpu_submission_recommended=recommendation,
            gpu_check_required=gpu_check_required,
            normalized_manifest=self.manifest.to_dict(),
            stages=stages,
            diagnostics=diagnostics,
            resource_estimate=resource_estimate,
            environment={
                "python_version": platform.python_version(),
                "python_implementation": platform.python_implementation(),
                "platform": platform.platform(),
                "torch_version": _distribution_version("torch"),
                "worker_python": sys.executable,
                "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
            },
        )
        if self.use_cache and status is not OverallStatus.INTERNAL_ERROR:
            cache.store(key, report)
        return report
