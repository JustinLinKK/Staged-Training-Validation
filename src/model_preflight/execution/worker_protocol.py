"""Versioned JSON-compatible controller/worker protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = 1
SHAPE_REDUCER_REVISION = 5


@dataclass(frozen=True)
class WorkerRequest:
    stage: str
    manifest: dict[str, Any]
    scenario: dict[str, Any] = field(default_factory=dict)
    shape_reduction_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "stage": self.stage,
            "manifest": self.manifest,
            "scenario": self.scenario,
            "shape_reduction_allowed": self.shape_reduction_allowed,
        }


@dataclass(frozen=True)
class WorkerResponse:
    status: str
    diagnostics: list[dict[str, Any]]
    evidence: dict[str, Any]
    captured_stdout: str = ""
    captured_stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "status": self.status,
            "diagnostics": self.diagnostics,
            "evidence": self.evidence,
            "captured_stdout": self.captured_stdout,
            "captured_stderr": self.captured_stderr,
        }

