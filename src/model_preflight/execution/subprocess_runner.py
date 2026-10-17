"""Fresh-process worker execution with bounded resources and output."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from model_preflight.core.diagnostics import Diagnostic
from model_preflight.core.enums import Classification, Severity, StageStatus
from model_preflight.core.results import StageResult
from model_preflight.execution.timeout import kill_process_group, terminate_process_group
from model_preflight.execution.worker_protocol import PROTOCOL_VERSION, WorkerRequest
from model_preflight.manifest.models import ExecutionConfig
from model_preflight.reporting.stacktrace import compact_stacktrace

_SECRET_FRAGMENTS = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY", "AUTH")


def _safe_environment(
    temp_dir: Path, disable_network: bool, cpu_threads: int = 1,
) -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not any(fragment in key.upper() for fragment in _SECRET_FRAGMENTS)
    }
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HOME": str(temp_dir),
            "TMPDIR": str(temp_dir),
            "XDG_CACHE_HOME": str(temp_dir / "cache"),
            "HF_HUB_OFFLINE": "1" if disable_network else "0",
            "TRANSFORMERS_OFFLINE": "1" if disable_network else "0",
            "OMP_NUM_THREADS": str(cpu_threads),
            "MKL_NUM_THREADS": str(cpu_threads),
            "OPENBLAS_NUM_THREADS": str(cpu_threads),
            "NUMEXPR_NUM_THREADS": str(cpu_threads),
            "VECLIB_MAXIMUM_THREADS": str(cpu_threads),
            "MODEL_PREFLIGHT_DISABLE_NETWORK": "1" if disable_network else "0",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _diagnostic(
    *,
    stage: str,
    code: str,
    message: str,
    classification: Classification,
    exception_type: str | None = None,
    stack_trace: str | None = None,
) -> Diagnostic:
    return Diagnostic(
        code=code,
        severity=(
            Severity.WARNING
            if classification is Classification.INCONCLUSIVE
            else Severity.ERROR
        ),
        stage=stage,
        classification=classification,
        message=message,
        exception_type=exception_type,
        stack_trace=compact_stacktrace(stack_trace or "") or None,
        reproduction=f"model-preflight check preflight.yaml --only {stage}",
    )


def _status(value: str) -> StageStatus:
    try:
        return StageStatus(value)
    except ValueError:
        return StageStatus.INTERNAL_ERROR


class SubprocessRunner:
    """Run one worker request in one disposable process."""

    def __init__(self, execution: ExecutionConfig) -> None:
        self.execution = execution

    def run(self, request: WorkerRequest, timeout_seconds: float) -> StageResult:
        started = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="model-preflight-") as temp_name:
            temp_dir = Path(temp_name)
            (temp_dir / "cache").mkdir()
            command = [sys.executable, "-m", "model_preflight.execution.worker_main"]

            environment = _safe_environment(
                temp_dir, self.execution.disable_network, self.execution.cpu_threads,
            )
            if self.execution.seed is not None:
                environment["PYTHONHASHSEED"] = str(self.execution.seed)
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=temp_dir,
                env=environment,
                start_new_session=True,
            )
            payload = json.dumps({
                **request.to_dict(),
                "resource_limits": {
                    "timeout_seconds": timeout_seconds * self.execution.cpu_threads,
                    "memory_mb": self.execution.maximum_cpu_memory_mb,
                    "maximum_processes": self.execution.maximum_processes,
                    "maximum_output_bytes": self.execution.maximum_output_bytes,
                },
                "cpu_threads": self.execution.cpu_threads,
            }, sort_keys=True)
            try:
                stdout, stderr = process.communicate(payload, timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                terminate_process_group(process.pid)
                try:
                    stdout, stderr = process.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    kill_process_group(process.pid)
                    stdout, stderr = process.communicate()
                diagnostic = _diagnostic(
                    stage=request.stage,
                    code="CHK002",
                    message=f"worker exceeded the {timeout_seconds:g}s wall-time limit",
                    classification=Classification.INCONCLUSIVE,
                    exception_type="TimeoutExpired",
                    stack_trace=stderr,
                )
                return StageResult(
                    name=request.stage,
                    status=StageStatus.INCONCLUSIVE,
                    duration_seconds=time.monotonic() - started,
                    diagnostics=[diagnostic],
                    evidence={"termination_cause": "wall_timeout"},
                )

            duration = time.monotonic() - started
            if process.returncode != 0:
                signal_number = -process.returncode if process.returncode < 0 else None
                resource_signal = signal_number in {
                    signal.SIGKILL,
                    signal.SIGXCPU,
                    signal.SIGXFSZ,
                }
                diagnostic = _diagnostic(
                    stage=request.stage,
                    code="CHK001" if resource_signal else "CHK003",
                    message=(
                        "worker was terminated by a resource limit"
                        if resource_signal
                        else f"worker exited unexpectedly with status {process.returncode}"
                    ),
                    classification=(
                        Classification.INCONCLUSIVE
                        if resource_signal
                        else Classification.CHECKER_ERROR
                    ),
                    stack_trace=stderr,
                )
                return StageResult(
                    name=request.stage,
                    status=(
                        StageStatus.INCONCLUSIVE
                        if resource_signal
                        else StageStatus.INTERNAL_ERROR
                    ),
                    duration_seconds=duration,
                    diagnostics=[diagnostic],
                    evidence={
                        "termination_cause": (
                            "resource_limit" if resource_signal else "worker_crash"
                        ),
                        "exit_status": process.returncode,
                        "signal": signal_number,
                    },
                )
            try:
                response: dict[str, Any] = json.loads(stdout)
                if response.get("protocol_version") != PROTOCOL_VERSION:
                    raise ValueError("unsupported worker protocol version")
                diagnostics = [
                    Diagnostic.from_dict(item) for item in response.get("diagnostics", [])
                ]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                diagnostic = _diagnostic(
                    stage=request.stage,
                    code="CHK003",
                    message=f"worker returned an invalid protocol response: {exc}",
                    classification=Classification.CHECKER_ERROR,
                    exception_type=type(exc).__name__,
                    stack_trace=stderr or stdout,
                )
                return StageResult(
                    name=request.stage,
                    status=StageStatus.INTERNAL_ERROR,
                    duration_seconds=duration,
                    diagnostics=[diagnostic],
                    evidence={"termination_cause": "protocol_error"},
                )
            evidence = dict(response.get("evidence", {}))
            if response.get("captured_stdout"):
                evidence["captured_stdout"] = str(response["captured_stdout"])
            if response.get("captured_stderr"):
                evidence["captured_stderr"] = str(response["captured_stderr"])
            return StageResult(
                name=request.stage,
                status=_status(str(response.get("status", "INTERNAL_ERROR"))),
                duration_seconds=duration,
                diagnostics=diagnostics,
                evidence=evidence,
            )
