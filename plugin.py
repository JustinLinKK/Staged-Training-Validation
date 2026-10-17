"""Agent-neutral static-check protocol and speculative two-epoch admission.

The light image checks source, not arbitrary training environments. Agents
provide a CPU probe command from their own candidate environment. A probe is
accepted only if it exits cleanly and reports exactly two completed epochs.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .preflight import check_source


def _top_level_imports(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        name.split(".", 1)[0]
        for node in tree.body
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
            else []
        )
    }


def check_request(request: dict[str, Any]) -> dict[str, Any]:
    """One JSON request to one JSON report; never run candidate source."""
    if request.get("version") != 1 or not isinstance(request.get("source"), str):
        raise ValueError("expected version=1 and string source")
    working_directory = request.get("working_directory")
    if working_directory is not None and not Path(working_directory).is_dir():
        raise ValueError("working_directory does not exist")
    modules = request.get("available_modules")
    if modules is not None and (not isinstance(modules, list) or not all(isinstance(x, str) for x in modules)):
        raise ValueError("available_modules must be a list of strings")
    shared_memory_bytes = request.get("shared_memory_bytes")
    if shared_memory_bytes is not None and (type(shared_memory_bytes) is not int or shared_memory_bytes < 0):
        raise ValueError("shared_memory_bytes must be a nonnegative integer")
    report = check_source(
        request["source"],
        filename=str(request.get("filename") or "<candidate>"),
        execution_timeout_seconds=request.get("execution_timeout_seconds", 3600),
        working_directory=working_directory,
        available_modules=set(modules) if modules is not None else None,
        shared_memory_bytes=shared_memory_bytes,
    )
    return {"version": 1, **report}


class DockerStaticChecker:
    """Run the small static checker without network, GPU, or writable mounts."""

    def __init__(self, image: str = "lightning-preflight:local", timeout_seconds: int = 60):
        self.image = image
        self.timeout_seconds = timeout_seconds

    def __call__(
        self, source: str, *, filename: str = "<candidate>",
        execution_timeout_seconds: int | None = 3600,
        working_directory: str | os.PathLike[str] | None = None,
    ) -> dict[str, Any]:
        root = Path(working_directory).resolve() if working_directory is not None else None
        command = [
            "docker", "run", "--rm", "-i", "--network", "none", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "64", "--memory", "512m", "--cpus", "1",
        ]
        if root is not None:
            command += ["--mount", f"type=bind,src={root},dst=/candidate,readonly"]
        command += [self.image]
        available = []
        for module in _top_level_imports(source):
            try:
                if importlib.util.find_spec(module) is not None:
                    available.append(module)
            except (ImportError, ValueError, AttributeError):
                pass
        request = {
            "version": 1, "source": source, "filename": filename,
            "execution_timeout_seconds": execution_timeout_seconds,
            "working_directory": "/candidate" if root is not None else None,
            "available_modules": available,
        }
        try:
            stats = os.statvfs("/dev/shm")
            request["shared_memory_bytes"] = stats.f_bavail * stats.f_frsize
        except OSError:
            pass
        completed = subprocess.run(
            command, input=json.dumps(request), text=True, capture_output=True,
            timeout=self.timeout_seconds, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"preflight container failed ({completed.returncode}): {completed.stderr[-1000:]}")
        report = json.loads(completed.stdout)
        if report.get("version") != 1 or not isinstance(report.get("admitted"), bool):
            raise RuntimeError("preflight container returned an invalid report")
        report["container_limits"] = [
            "third-party library registry checks require libraries installed in the checker image",
            "hardware-specific checks requiring nvidia-smi are unavailable in the CPU-only image",
        ]
        return report


@dataclass(frozen=True)
class ProbeSpec:
    argv: tuple[str, ...]
    cwd: Path
    timeout_seconds: int = 300


def probe_from_environment(script_path: Path, working_directory: Path, node_id: str) -> ProbeSpec:
    """Resolve a trusted adapter command; never invoke a shell or candidate text."""
    raw = os.environ.get("LIGHTNING_PREFLIGHT_CPU_PROBE_ARGV")
    if not raw:
        raise ValueError("LIGHTNING_PREFLIGHT_CPU_PROBE_ARGV is required")
    try:
        template = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("CPU probe command must be a JSON argv array") from exc
    if not isinstance(template, list) or not template or not all(isinstance(item, str) for item in template):
        raise ValueError("CPU probe command must be a nonempty JSON argv array")
    if not any("{script_path}" in item for item in template):
        raise ValueError("CPU probe command must include {script_path}")
    if not any("{epochs}" in item for item in template):
        raise ValueError("CPU probe command must include {epochs}")
    replacements = {
        "{script_path}": str(Path(script_path).resolve()),
        "{working_directory}": str(Path(working_directory).resolve()),
        "{node_id}": str(node_id),
        "{epochs}": "2",
    }
    argv = []
    for item in template:
        for key, value in replacements.items():
            item = item.replace(key, value)
        argv.append(item)
    timeout = int(os.environ.get("LIGHTNING_PREFLIGHT_CPU_PROBE_TIMEOUT", "300"))
    return ProbeSpec(tuple(argv), Path(working_directory).resolve(), timeout)


def _cpu_probe(spec: ProbeSpec) -> dict[str, Any]:
    if not spec.argv or spec.timeout_seconds <= 0:
        return {"status": "failed", "reason": "invalid_probe_spec"}
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "-1", "NVIDIA_VISIBLE_DEVICES": "void",
        "LIGHTNING_CPU_EPOCHS": "2",
    })
    process = subprocess.Popen(
        spec.argv, cwd=spec.cwd, env=environment, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=spec.timeout_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return {"status": "failed", "reason": "timeout"}
    if process.returncode != 0:
        return {"status": "failed", "reason": "nonzero_exit", "exit_code": process.returncode,
                "stderr_tail": stderr[-2000:]}
    try:
        evidence = json.loads(stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"status": "failed", "reason": "missing_epoch_evidence", "stdout_tail": stdout[-2000:]}
    if isinstance(evidence, dict) and evidence.get("status") == "failed" and evidence.get("bug_confirmed") is True:
        return {"status": "confirmed_bug", "epochs_completed": evidence.get("epochs_completed"), "evidence": evidence}
    if not isinstance(evidence, dict) or evidence.get("status") != "passed" or evidence.get("epochs_completed") != 2:
        return {"status": "inconclusive", "reason": "incomplete_or_failed_epochs", "evidence": evidence}
    return {"status": "passed", "epochs_completed": 2, "evidence": evidence}


class SpeculativePreflight:
    """Static gate, provisional scheduler submission, then bounded CPU proof."""

    def __init__(self, static_check: Callable[..., dict[str, Any]] = check_source):
        self.static_check = static_check

    def run(
        self, source: str, *, workspace: Path, node_id: str,
        submit: Callable[[], str], cancel: Callable[[str], Any], probe: ProbeSpec,
        execution_timeout_seconds: int | None = 3600,
        static_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        report_dir = Path(workspace) / "preflight" / str(node_id)
        report_dir.mkdir(parents=True, exist_ok=True)
        report = static_report if static_report is not None else self.static_check(
            source, filename=f"candidate-{node_id}.py",
            execution_timeout_seconds=execution_timeout_seconds,
            working_directory=workspace,
        )
        (report_dir / "static.json").write_text(json.dumps(report, indent=2) + "\n")
        if not report["admitted"]:
            (report_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            outcome = {"status": "static_rejected", "static": report, "job_id": None}
        else:
            job_id = submit()
            try:
                evidence = _cpu_probe(probe)
            except (OSError, ValueError) as exc:
                evidence = {"status": "failed", "reason": type(exc).__name__, "detail": str(exc)}
            if evidence["status"] == "passed":
                outcome = {"status": "probe_passed", "static": report, "job_id": job_id, "probe": evidence}
            elif evidence["status"] == "confirmed_bug":
                final_report = {
                    "status": "rejected", "admitted": False,
                    "assurance": "two_epoch_cpu_probe", "issues": ["cpu_probe_confirmed_bug"],
                    "diagnostics": [{"code": "cpu_probe_confirmed_bug",
                                     "message": "The candidate CPU probe reproduced a defect",
                                     "evidence": evidence}],
                }
                (report_dir / "report.json").write_text(json.dumps(final_report, indent=2) + "\n")
                cancel(job_id)
                outcome = {"status": "probe_rejected", "static": report, "job_id": job_id, "probe": evidence}
            else:
                outcome = {"status": "probe_inconclusive", "static": report, "job_id": job_id, "probe": evidence}
        (report_dir / "decision.json").write_text(json.dumps(outcome, indent=2) + "\n")
        return outcome


def submit_with_speculation(
    client: Any, job: Any, source: str, script_path: Path,
    workspace: Path, node_id: str, static_report: dict[str, Any] | None,
    execution_timeout_seconds: int | None = 3600,
) -> tuple[Any, dict[str, Any] | None]:
    """Drop-in scheduler bridge; old admission remains unchanged unless enabled."""
    if os.environ.get("LIGHTNING_PREFLIGHT_SPECULATIVE") != "1":
        return client.submit(job), None
    if static_report is None or not static_report.get("admitted"):
        raise ValueError("speculative admission requires a successful static report")
    probe = probe_from_environment(script_path, workspace, node_id)
    submitted: list[Any] = []

    def submit() -> str:
        submitted_job = client.submit(job)
        submitted.append(submitted_job)
        return submitted_job.job_id

    outcome = SpeculativePreflight().run(
        source, workspace=workspace, node_id=node_id,
        submit=submit, cancel=client.cancel, probe=probe,
        static_report=static_report,
        execution_timeout_seconds=execution_timeout_seconds,
    )
    return submitted[0], outcome


def main() -> int:
    argparse.ArgumentParser(description="Lightning preflight JSON protocol v1").parse_args()
    try:
        report = check_request(json.load(sys.stdin))
    except (ValueError, TypeError, KeyError) as exc:
        print(json.dumps({"version": 1, "status": "invalid_request", "error": str(exc)}))
        return 2
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
