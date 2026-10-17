"""Typed normalized manifest models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from model_preflight.core.enums import PolicyMode


@dataclass(frozen=True)
class CandidateConfig:
    id: str
    root: Path
    adapter: str


@dataclass(frozen=True)
class TaskConfig:
    name: str
    fixture_root: Path | None = None
    num_classes: int | None = None
    target_dtype: str | None = None
    input_rank: int | None = None
    output_class_dimension: int | None = None
    drop_last: bool = False


@dataclass(frozen=True)
class ScenarioConfig:
    train_batch_sizes: tuple[int, ...]
    test_last_batch: bool = True
    last_batch_size: int = 1
    run_validation: bool = True
    input_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    boundary_shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    fixture: dict[str, tuple[int, ...]] = field(default_factory=dict)
    precision: tuple[str, ...] = ("fp32",)
    stateful_two_steps: bool = False

    def matrix(self) -> list[dict[str, Any]]:
        shapes = self.input_shapes | self.boundary_shapes
        shape_items: list[tuple[str, tuple[int, ...]]]
        shape_items = list(shapes.items()) if shapes else [("default", ())]
        result: list[dict[str, Any]] = []
        for batch_size in self.train_batch_sizes:
            for shape_name, shape in shape_items:
                for precision in self.precision:
                    result.append(
                        {
                            "batch_size": batch_size,
                            "shape_name": shape_name,
                            "input_shape": list(shape),
                            "precision": precision,
                            "mode": "train",
                            "fixture": {key: list(value) for key, value in self.fixture.items()},
                        }
                    )
        return result

    def cpu_scenarios(self) -> list[dict[str, Any]]:
        first_shape = next(iter(self.input_shapes.items()), ("default", ()))
        normal = {
            "batch_size": self.train_batch_sizes[0],
            "shape_name": first_shape[0],
            "input_shape": list(first_shape[1]),
            "precision": "fp32",
            "mode": "train",
            "fixture": {key: list(value) for key, value in self.fixture.items()},
        }
        result = [normal]
        if self.test_last_batch and self.last_batch_size != normal["batch_size"]:
            result.append(
                {
                    **normal,
                    "batch_size": self.last_batch_size,
                    "shape_name": "last_batch",
                }
            )
        return result


@dataclass(frozen=True)
class TargetConfig:
    profile: str


@dataclass(frozen=True)
class ShapeReductionConfig:
    mode: str = "off"
    width_factor: float = 0.5
    minimum_width: int = 16


@dataclass(frozen=True)
class ExecutionConfig:
    abstract_timeout_seconds: float = 30.0
    cpu_timeout_seconds: float = 90.0
    maximum_cpu_memory_mb: int = 8192
    maximum_processes: int = 32
    maximum_output_bytes: int = 1_000_000
    disable_network: bool = True
    allow_real_cpu_abstract_fallback: bool = False
    cache: bool = False
    shape_reduction: ShapeReductionConfig = field(default_factory=ShapeReductionConfig)
    seed: int | None = None


@dataclass(frozen=True)
class PolicyConfig:
    mode: PolicyMode = PolicyMode.BALANCED
    memory_warning_fraction: float = 0.7
    memory_high_risk_fraction: float = 0.9
    memory_failure_fraction: float | None = None


@dataclass(frozen=True)
class Manifest:
    schema_version: int
    candidate: CandidateConfig
    task: TaskConfig
    scenarios: ScenarioConfig
    target: TargetConfig
    execution: ExecutionConfig
    policy: PolicyConfig
    source_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate": {
                "id": self.candidate.id,
                "root": str(self.candidate.root),
                "adapter": self.candidate.adapter,
            },
            "task": {
                key: value
                for key, value in {
                    "name": self.task.name,
                    "fixture_root": (
                        str(self.task.fixture_root) if self.task.fixture_root is not None else None
                    ),
                    "num_classes": self.task.num_classes,
                    "target_dtype": self.task.target_dtype,
                    "input_rank": self.task.input_rank,
                    "output_class_dimension": self.task.output_class_dimension,
                    "drop_last": self.task.drop_last,
                }.items()
                if value is not None
            },
            "scenarios": {
                "train_batch_sizes": list(self.scenarios.train_batch_sizes),
                "test_last_batch": self.scenarios.test_last_batch,
                "last_batch_size": self.scenarios.last_batch_size,
                "run_validation": self.scenarios.run_validation,
                "input_shapes": {
                    key: list(value) for key, value in self.scenarios.input_shapes.items()
                },
                "boundary_shapes": {
                    key: list(value) for key, value in self.scenarios.boundary_shapes.items()
                },
                "fixture": {
                    key: list(value) for key, value in self.scenarios.fixture.items()
                },
                "precision": list(self.scenarios.precision),
                "stateful_two_steps": self.scenarios.stateful_two_steps,
            },
            "target": {"profile": self.target.profile},
            "execution": {
                "abstract_timeout_seconds": self.execution.abstract_timeout_seconds,
                "cpu_timeout_seconds": self.execution.cpu_timeout_seconds,
                "maximum_cpu_memory_mb": self.execution.maximum_cpu_memory_mb,
                "maximum_processes": self.execution.maximum_processes,
                "maximum_output_bytes": self.execution.maximum_output_bytes,
                "disable_network": self.execution.disable_network,
                "allow_real_cpu_abstract_fallback": (
                    self.execution.allow_real_cpu_abstract_fallback
                ),
                "cache": self.execution.cache,
                "shape_reduction": {
                    "mode": self.execution.shape_reduction.mode,
                    "width_factor": self.execution.shape_reduction.width_factor,
                    "minimum_width": self.execution.shape_reduction.minimum_width,
                },
                **({"seed": self.execution.seed} if self.execution.seed is not None else {}),
            },
            "policy": {
                key: value
                for key, value in {
                    "mode": self.policy.mode.value,
                    "memory_warning_fraction": self.policy.memory_warning_fraction,
                    "memory_high_risk_fraction": self.policy.memory_high_risk_fraction,
                    "memory_failure_fraction": self.policy.memory_failure_fraction,
                }.items()
                if value is not None
            },
        }
