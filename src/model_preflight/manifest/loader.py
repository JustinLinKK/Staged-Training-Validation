"""YAML manifest loader with JSON Schema and semantic validation."""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any, cast

import jsonschema
import yaml

from model_preflight.core.enums import PolicyMode
from model_preflight.core.exceptions import ManifestError
from model_preflight.manifest.models import (
    CandidateConfig,
    ExecutionConfig,
    Manifest,
    PolicyConfig,
    ScenarioConfig,
    ShapeReductionConfig,
    TargetConfig,
    TaskConfig,
)
from model_preflight.manifest.validator import validate_paths, validate_thresholds


def _schema() -> dict[str, Any]:
    resource = files("model_preflight.manifest").joinpath("schema_v1.json")
    return cast(dict[str, Any], json.loads(resource.read_text(encoding="utf-8")))


def _path(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value).expanduser()
    return (base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def load_manifest(path: str | Path) -> Manifest:
    source_path = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {source_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ManifestError(f"invalid YAML in {source_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError("manifest root must be a mapping")
    if raw.get("schema_version") != 1:
        raise ManifestError(
            f"unsupported manifest schema_version: {raw.get('schema_version')!r}; expected 1"
        )
    try:
        jsonschema.Draft202012Validator(_schema()).validate(raw)
    except jsonschema.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
        raise ManifestError(f"manifest validation failed at {location}: {exc.message}") from exc

    base = source_path.parent
    candidate_raw = raw["candidate"]
    task_raw = raw["task"]
    scenario_raw = raw["scenarios"]
    execution_raw = raw.get("execution", {})
    policy_raw = raw.get("policy", {})
    candidate_root = _path(base, candidate_raw["root"])
    assert candidate_root is not None
    fixture_root = _path(base, task_raw.get("fixture_root"))
    validate_paths(candidate_root, fixture_root)

    warning = float(policy_raw.get("memory_warning_fraction", 0.7))
    high_risk = float(policy_raw.get("memory_high_risk_fraction", 0.9))
    failure_raw = policy_raw.get("memory_failure_fraction")
    failure = float(failure_raw) if failure_raw is not None else None
    validate_thresholds(warning, high_risk, failure)

    return Manifest(
        schema_version=1,
        candidate=CandidateConfig(
            id=candidate_raw["id"],
            root=candidate_root,
            adapter=candidate_raw["adapter"],
        ),
        task=TaskConfig(
            name=task_raw["name"],
            fixture_root=fixture_root,
            num_classes=task_raw.get("num_classes"),
            target_dtype=task_raw.get("target_dtype"),
            input_rank=task_raw.get("input_rank"),
            output_class_dimension=task_raw.get("output_class_dimension"),
            drop_last=bool(task_raw.get("drop_last", False)),
        ),
        scenarios=ScenarioConfig(
            train_batch_sizes=tuple(scenario_raw["train_batch_sizes"]),
            test_last_batch=bool(scenario_raw.get("test_last_batch", True)),
            last_batch_size=int(scenario_raw.get("last_batch_size", 1)),
            run_validation=bool(scenario_raw.get("run_validation", True)),
            input_shapes={
                key: tuple(value) for key, value in scenario_raw.get("input_shapes", {}).items()
            },
            boundary_shapes={
                key: tuple(value) for key, value in scenario_raw.get("boundary_shapes", {}).items()
            },
            fixture={
                key: tuple(value) for key, value in scenario_raw.get("fixture", {}).items()
            },
            precision=tuple(scenario_raw.get("precision", ["fp32"])),
            stateful_two_steps=bool(scenario_raw.get("stateful_two_steps", False)),
        ),
        target=TargetConfig(profile=raw["target"]["profile"]),
        execution=ExecutionConfig(
            abstract_timeout_seconds=float(
                execution_raw.get("abstract_timeout_seconds", 30.0)
            ),
            cpu_timeout_seconds=float(execution_raw.get("cpu_timeout_seconds", 90.0)),
            maximum_cpu_memory_mb=int(execution_raw.get("maximum_cpu_memory_mb", 8192)),
            maximum_processes=int(execution_raw.get("maximum_processes", 32)),
            maximum_output_bytes=int(execution_raw.get("maximum_output_bytes", 1_000_000)),
            disable_network=bool(execution_raw.get("disable_network", True)),
            allow_real_cpu_abstract_fallback=bool(
                execution_raw.get("allow_real_cpu_abstract_fallback", False)
            ),
            cache=bool(execution_raw.get("cache", False)),
            shape_reduction=ShapeReductionConfig(**execution_raw.get("shape_reduction", {})),
            seed=execution_raw.get("seed"),
            max_workers=int(execution_raw.get("max_workers", 1)),
            cpu_threads=int(execution_raw.get("cpu_threads", 1)),
        ),
        policy=PolicyConfig(
            mode=PolicyMode(policy_raw.get("mode", "balanced")),
            memory_warning_fraction=warning,
            memory_high_risk_fraction=high_risk,
            memory_failure_fraction=failure,
        ),
        source_path=source_path,
    )
