"""Disposable worker entry point.

This module intentionally imports no PyTorch or candidate modules until after the
controller request is parsed and stdout/stderr are redirected away from the JSON
protocol channel.
"""

from __future__ import annotations

import errno
import json
import os
import random
import resource
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Thread
from typing import Any

from model_preflight.core.diagnostics import Diagnostic
from model_preflight.core.enums import Classification, Severity, StageName, StageStatus
from model_preflight.execution.resource_limits import apply_resource_limits
from model_preflight.execution.worker_protocol import PROTOCOL_VERSION, WorkerResponse
from model_preflight.reporting.stacktrace import compact_stacktrace


@dataclass
class ConfirmedFailure(Exception):
    code: str
    message: str
    evidence: dict[str, Any]


@dataclass
class CandidateRaised(Exception):
    area: str
    original: BaseException
    trace: str


def _candidate_call(area: str, function: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except BaseException as exc:
        raise CandidateRaised(area, exc, traceback.format_exc()) from exc


def _block_network(event: str, _args: tuple[Any, ...]) -> None:
    if event in {"socket.connect", "socket.bind"}:
        raise PermissionError("network access is disabled by Model Preflight")


def _context(manifest: dict[str, Any], scenario: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        "manifest": manifest,
        "scenario": scenario,
        "device": device,
        "fixture_root": manifest["task"].get("fixture_root"),
    }


def _load(manifest: dict[str, Any]) -> Any:
    from model_preflight.adapters.pytorch import load_adapter

    return _candidate_call(
        "import",
        load_adapter,
        Path(manifest["candidate"]["root"]),
        manifest["candidate"]["adapter"],
    )


def _construct(
    manifest: dict[str, Any],
    scenario: dict[str, Any],
    *,
    optimizer: bool,
    device: str = "cpu",
    runtime_evidence: dict[str, Any] | None = None,
    reduction_allowed: bool = False,
) -> tuple[Any, Any, Any | None, dict[str, Any]]:
    import torch

    adapter = _load(manifest)
    context = _context(manifest, scenario, device)
    model = _candidate_call("construction", adapter.build_model, context)
    if not isinstance(model, torch.nn.Module):
        raise ConfirmedFailure(
            "CON001",
            f"build_model returned {type(model).__name__}, expected torch.nn.Module",
            {"returned_type": type(model).__name__},
        )
    if runtime_evidence is not None:
        settings = manifest["execution"].get("shape_reduction", {"mode": "off"})
        if reduction_allowed and settings["mode"] == "auto":
            from model_preflight.execution.shape_reduction import reduce_model

            runtime_evidence["shape_reduction"] = reduce_model(model, adapter, context, settings)
        else:
            runtime_evidence["shape_reduction"] = {
                "applied": False,
                "execution_model": "original",
                "reason": "off" if settings["mode"] == "off" else "prerequisites_not_passed",
            }
    parameters = list(model.parameters())
    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise ConfirmedFailure(
            "OPT001",
            "model contains no trainable parameters",
            {"parameter_tensors": len(parameters)},
        )
    result_optimizer = None
    if optimizer:
        result_optimizer = _candidate_call(
            "optimizer", adapter.build_optimizer, model, context
        )
        if not isinstance(result_optimizer, torch.optim.Optimizer):
            raise ConfirmedFailure(
                "OPT001",
                "build_optimizer did not return torch.optim.Optimizer",
                {"returned_type": type(result_optimizer).__name__},
            )
        optimizer_parameters = [
            parameter
            for group in result_optimizer.param_groups
            for parameter in group.get("params", [])
        ]
        if not optimizer_parameters:
            raise ConfirmedFailure("OPT001", "optimizer contains no parameters", {})
        model_ids = {id(parameter) for parameter in parameters}
        foreign = sum(id(parameter) not in model_ids for parameter in optimizer_parameters)
        if foreign:
            raise ConfirmedFailure(
                "OPT002",
                "optimizer contains parameters that do not belong to the model",
                {"foreign_parameter_tensors": foreign},
            )
    evidence = {
        "model_type": f"{type(model).__module__}.{type(model).__qualname__}",
        "parameter_tensors": len(parameters),
        "trainable_parameter_tensors": len(trainable),
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in parameters
        ),
        "trainable_parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in trainable
        ),
    }
    if result_optimizer is not None:
        evidence["optimizer_type"] = (
            f"{type(result_optimizer).__module__}.{type(result_optimizer).__qualname__}"
        )
    return adapter, model, result_optimizer, evidence


def _describe(value: Any, depth: int = 0) -> Any:
    import torch

    if depth > 5:
        return {"type": type(value).__name__, "truncated": True}
    if isinstance(value, torch.Tensor):
        return {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "device": str(value.device),
            "requires_grad": value.requires_grad,
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": {
                str(key): _describe(item, depth + 1)
                for key, item in list(value.items())[:32]
            },
        }
    if isinstance(value, (tuple, list)):
        return {
            "type": type(value).__name__,
            "items": [_describe(item, depth + 1) for item in value[:32]],
        }
    return {"type": type(value).__name__, "repr": repr(value)[:200]}


def _find_inputs_and_target(batch: Any) -> tuple[Any | None, Any | None]:
    import torch

    if isinstance(batch, torch.Tensor):
        return batch, None
    if isinstance(batch, (tuple, list)):
        return (
            batch[0] if batch else None,
            batch[1] if len(batch) > 1 else None,
        )
    if isinstance(batch, dict):
        inputs = next(
            (
                batch[key]
                for key in (
                    "inputs",
                    "input",
                    "x",
                    "images",
                    "image",
                    "pixel_values",
                    "features",
                    "feats",
                )
                if key in batch
            ),
            None,
        )
        target = next(
            (
                batch[key]
                for key in ("targets", "target", "labels", "label", "y")
                if key in batch
            ),
            None,
        )
        return inputs, target
    return None, None


def _output_tensor(value: Any) -> Any | None:
    import torch

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next(
            (item for item in value if isinstance(item, torch.Tensor)),
            None,
        )
    if isinstance(value, dict):
        for key in ("logits", "output", "outputs", "prediction", "predictions"):
            if isinstance(value.get(key), torch.Tensor):
                return value[key]
    return None


def _validate_output_contract(output: Any, manifest: dict[str, Any]) -> None:
    task = manifest["task"]
    dimension = task.get("output_class_dimension")
    classes = task.get("num_classes")
    if dimension is None or classes is None:
        return
    tensor = _output_tensor(output)
    if tensor is None:
        return
    normalized_dimension = dimension if dimension >= 0 else tensor.ndim + dimension
    if normalized_dimension < 0 or normalized_dimension >= tensor.ndim:
        raise ConfirmedFailure(
            "OUT001",
            f"configured output class dimension {dimension} is invalid for rank {tensor.ndim}",
            {"output_shape": list(tensor.shape), "output_class_dimension": dimension},
        )
    actual = tensor.shape[normalized_dimension]
    if actual != classes:
        raise ConfirmedFailure(
            "OUT001",
            f"model output has {actual} classes on dimension {dimension}, expected {classes}",
            {
                "output_shape": list(tensor.shape),
                "output_class_dimension": dimension,
                "expected_num_classes": classes,
            },
        )


def _build_batch(
    adapter: Any,
    manifest: dict[str, Any],
    scenario: dict[str, Any],
    device: str,
    *,
    validation: bool = False,
) -> Any:
    method_name = "build_validation_batch" if validation else "build_train_batch"
    method = getattr(adapter, method_name, None)
    if not callable(method):
        raise ConfirmedFailure(
            "VAL001" if validation else "DAT001",
            f"adapter does not define {method_name}",
            {},
        )
    return _candidate_call("fixture", method, scenario, device)


def _validate_batch(batch: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    import torch

    inputs, target = _find_inputs_and_target(batch)
    task = manifest["task"]
    if inputs is None:
        raise ConfirmedFailure(
            "DAT001",
            "could not identify model inputs in the representative batch",
            {"batch_type": type(batch).__name__},
        )
    expected_rank = task.get("input_rank")
    if expected_rank is not None and isinstance(inputs, torch.Tensor):
        actual_rank = max(0, inputs.ndim - 1)
        if actual_rank != expected_rank:
            raise ConfirmedFailure(
                "SHP001",
                f"input rank excluding batch is {actual_rank}, expected {expected_rank}",
                {"actual_shape": list(inputs.shape), "expected_input_rank": expected_rank},
            )
    expected_dtype = task.get("target_dtype")
    if expected_dtype is not None:
        if not isinstance(target, torch.Tensor):
            raise ConfirmedFailure(
                "DAT002",
                "manifest declares target_dtype but no target tensor was found",
                {"expected_target_dtype": expected_dtype},
            )
        actual_dtype = str(target.dtype).removeprefix("torch.")
        if actual_dtype != expected_dtype:
            raise ConfirmedFailure(
                "DAT002",
                f"target dtype is {actual_dtype}, expected {expected_dtype}",
                {"actual_target_dtype": actual_dtype, "expected_target_dtype": expected_dtype},
            )
    return {"batch": _describe(batch)}


def _construct_stage(manifest: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    _, _, _, evidence = _construct(manifest, scenario, optimizer=True)
    return evidence


def _data_stage(manifest: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    adapter = _load(manifest)
    batch = _build_batch(adapter, manifest, scenario, "cpu")
    return _validate_batch(batch, manifest)


def _check_tensor_finite(value: Any, code: str, message: str) -> None:
    import torch

    if isinstance(value, dict):
        for item in value.values():
            _check_tensor_finite(item, code, message)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _check_tensor_finite(item, code, message)
    if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all().item()):
        raise ConfirmedFailure(
            code,
            message,
            {"dtype": str(value.dtype), "shape": list(value.shape)},
        )


def _cpu_training_stage(
    manifest: dict[str, Any], scenario: dict[str, Any],
    runtime_evidence: dict[str, Any] | None = None, reduction_allowed: bool = False,
) -> dict[str, Any]:
    import torch

    adapter, model, optimizer, evidence = _construct(
        manifest, scenario, optimizer=True, runtime_evidence=runtime_evidence,
        reduction_allowed=reduction_allowed,
    )
    assert optimizer is not None
    batch = _build_batch(adapter, manifest, scenario, "cpu")
    _validate_batch(batch, manifest)
    _candidate_call("training", model.train)
    steps = 2 if manifest["scenarios"].get("stateful_two_steps", False) else 1
    final_loss = None
    captured_outputs: list[Any] = []
    output_handle = model.register_forward_hook(
        lambda _module, _inputs, output: captured_outputs.append(output)
    )
    execution_started = time.monotonic()
    try:
        for _ in range(steps):
            captured_outputs.clear()
            optimizer.zero_grad(set_to_none=True)
            with torch.autograd.detect_anomaly(check_nan=True):
                loss = _candidate_call(
                    "training",
                    adapter.training_step,
                    model,
                    batch,
                    _context(manifest, scenario, "cpu"),
                )
                if captured_outputs:
                    _validate_output_contract(captured_outputs[-1], manifest)
                if not isinstance(loss, torch.Tensor):
                    raise ConfirmedFailure(
                        "LOS001",
                        "training_step must return a torch.Tensor loss",
                        {"returned_type": type(loss).__name__},
                    )
                if loss.numel() != 1:
                    raise ConfirmedFailure(
                        "LOS001",
                        "training_step must return a scalar loss tensor",
                        {"loss_shape": list(loss.shape)},
                    )
                _check_tensor_finite(loss, "NUM001", "training loss is non-finite")
                if not loss.requires_grad:
                    raise ConfirmedFailure(
                        "AUT001", "training loss does not require gradients", {}
                    )
                _candidate_call("backward", loss.backward)
            trainable = {
                name: parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
            with_grad = {
                name: parameter
                for name, parameter in trainable.items()
                if parameter.grad is not None
            }
            if not with_grad:
                raise ConfirmedFailure(
                    "AUT003",
                    "no trainable model parameter received a gradient",
                    {"trainable_parameters": sorted(trainable)},
                )
            required_method = getattr(adapter, "required_gradient_names", None)
            if callable(required_method):
                required = set(_candidate_call("training", required_method, model))
                missing = sorted(required - set(with_grad))
                if missing:
                    raise ConfirmedFailure(
                        "AUT003",
                        "required trainable parameters did not receive gradients",
                        {"missing_parameters": missing},
                    )
            for name, parameter in with_grad.items():
                assert parameter.grad is not None
                if not bool(torch.isfinite(parameter.grad).all().item()):
                    raise ConfirmedFailure(
                        "NUM002",
                        f"gradient for {name} is non-finite",
                        {"parameter": name, "shape": list(parameter.shape)},
                    )
            _candidate_call("optimizer_step", optimizer.step)
            for name, parameter in model.named_parameters():
                _check_tensor_finite(
                    parameter, "NUM001", f"parameter {name} is non-finite after optimizer step"
                )
            _check_tensor_finite(optimizer.state, "NUM001", "optimizer state is non-finite")
            final_loss = float(loss.detach().cpu().item())
    finally:
        output_handle.remove()
        if runtime_evidence is not None:
            runtime_evidence["workflow_seconds"] = time.monotonic() - execution_started
    evidence.update(
        {
            "loss": final_loss,
            "steps": steps,
            "batch": _describe(batch),
            "parameters_with_gradients": len(with_grad),
        }
    )
    return evidence


def _validation_stage(
    manifest: dict[str, Any], scenario: dict[str, Any],
    runtime_evidence: dict[str, Any] | None = None, reduction_allowed: bool = False,
) -> dict[str, Any]:
    import torch

    adapter, model, _, evidence = _construct(
        manifest, scenario, optimizer=False, runtime_evidence=runtime_evidence,
        reduction_allowed=reduction_allowed,
    )
    validation_step = getattr(adapter, "validation_step", None)
    if not callable(validation_step):
        raise ConfirmedFailure("VAL001", "adapter does not define validation_step", {})
    batch = _build_batch(adapter, manifest, scenario, "cpu", validation=True)
    _validate_batch(batch, manifest)
    _candidate_call("validation", model.eval)
    execution_started = time.monotonic()
    with torch.no_grad():
        output = _candidate_call(
            "validation",
            validation_step,
            model,
            batch,
            _context(manifest, scenario, "cpu"),
        )
    _check_tensor_finite(output, "NUM001", "validation output is non-finite")
    _validate_output_contract(output, manifest)
    evidence["batch"] = _describe(batch)
    evidence["validation_output"] = _describe(output)
    if runtime_evidence is not None:
        runtime_evidence["workflow_seconds"] = time.monotonic() - execution_started
    return evidence


def _export_inputs(batch: Any) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    import torch

    inputs, _ = _find_inputs_and_target(batch)
    if isinstance(inputs, torch.Tensor):
        return (inputs,), {}
    if isinstance(inputs, (tuple, list)) and all(
        isinstance(item, torch.Tensor) for item in inputs
    ):
        return tuple(inputs), {}
    return None


def _tensor_bytes(value: Any) -> int:
    import torch

    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    return 0


def _abstract_meta_stage(
    manifest: dict[str, Any], scenario: dict[str, Any]
) -> dict[str, Any]:
    import torch

    adapter = _load(manifest)
    context = _context(manifest, scenario, "meta")
    with torch.device("meta"):
        model = _candidate_call("construction", adapter.build_model, context)
    if not isinstance(model, torch.nn.Module):
        raise ConfirmedFailure(
            "CON001", "build_model did not return torch.nn.Module", {}
        )
    model = model.to("meta")
    batch = _build_batch(adapter, manifest, scenario, "meta")
    _validate_batch(batch, manifest)

    evidence: dict[str, Any] = {
        "backend": "meta",
        "batch": _describe(batch),
        "torch_export_attempted": False,
        "torch_export_succeeded": False,
    }
    export_inputs = _export_inputs(batch)
    if export_inputs is not None and hasattr(torch, "export"):
        evidence["torch_export_attempted"] = True
        try:
            torch.export.export(model, export_inputs[0], export_inputs[1])
            evidence["torch_export_succeeded"] = True
            evidence["backend"] = "torch.export+meta"
        except BaseException as exc:
            evidence["torch_export_error"] = f"{type(exc).__name__}: {exc}"[:1000]

    activation_bytes = 0
    captured_outputs: list[Any] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        nonlocal activation_bytes
        activation_bytes += _tensor_bytes(output)

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if module is not model and not any(module.children())
    ]
    output_handle = model.register_forward_hook(
        lambda _module, _inputs, output: captured_outputs.append(output)
    )
    try:
        loss = _candidate_call(
            "abstract",
            adapter.training_step,
            model,
            batch,
            context,
        )
    finally:
        output_handle.remove()
        for handle in handles:
            handle.remove()
    if captured_outputs:
        _validate_output_contract(captured_outputs[-1], manifest)
    if not isinstance(loss, torch.Tensor):
        raise ConfirmedFailure(
            "LOS001",
            "training_step did not return a tensor during abstract execution",
            {"returned_type": type(loss).__name__},
        )
    if loss.numel() != 1:
        raise ConfirmedFailure(
            "LOS001",
            "abstract loss is not scalar",
            {"loss_shape": list(loss.shape)},
        )
    evidence.update(
        {
            "loss": _describe(loss),
            "estimated_forward_activation_bytes": activation_bytes,
        }
    )
    return evidence


def _unsupported_abstract(raised: CandidateRaised) -> bool:
    message = str(raised.original).lower()
    return isinstance(raised.original, NotImplementedError) or any(
        term in message
        for term in (
            "no meta",
            "meta tensor",
            "fake tensor",
            "not implemented",
            "data-dependent",
        )
    )


def _abstract_real_cpu_stage(
    manifest: dict[str, Any], scenario: dict[str, Any]
) -> dict[str, Any]:
    import torch

    adapter, model, _, evidence = _construct(
        manifest, scenario, optimizer=False, device="cpu"
    )
    batch = _build_batch(adapter, manifest, scenario, "cpu")
    _validate_batch(batch, manifest)
    captured_outputs: list[Any] = []
    handle = model.register_forward_hook(
        lambda _module, _inputs, output: captured_outputs.append(output)
    )
    try:
        loss = _candidate_call(
            "abstract",
            adapter.training_step,
            model,
            batch,
            _context(manifest, scenario, "cpu"),
        )
    finally:
        handle.remove()
    if captured_outputs:
        _validate_output_contract(captured_outputs[-1], manifest)
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise ConfirmedFailure(
            "LOS001",
            "real CPU abstract fallback did not produce a scalar loss tensor",
            {"loss": _describe(loss)},
        )
    evidence.update(
        {
            "backend": "real_cpu_fallback",
            "batch": _describe(batch),
            "loss": _describe(loss),
            "estimated_forward_activation_bytes": 0,
            "torch_export_attempted": False,
            "torch_export_succeeded": False,
        }
    )
    return evidence


def _abstract_fake_stage(
    manifest: dict[str, Any], scenario: dict[str, Any]
) -> dict[str, Any]:
    import torch

    from model_preflight.execution.torch_compat import create_fake_tensor_mode

    mode = create_fake_tensor_mode()
    if mode is None:
        raise CandidateRaised(
            "abstract",
            NotImplementedError("FakeTensorMode is unavailable in this PyTorch version"),
            "",
        )
    adapter = _load(manifest)
    context = _context(manifest, scenario, "cpu")
    activation_bytes = 0
    captured_outputs: list[Any] = []
    with mode:
        model = _candidate_call("construction", adapter.build_model, context)
        if not isinstance(model, torch.nn.Module):
            raise ConfirmedFailure(
                "CON001", "build_model did not return torch.nn.Module", {}
            )
        batch = _build_batch(adapter, manifest, scenario, "cpu")
        _validate_batch(batch, manifest)

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            nonlocal activation_bytes
            activation_bytes += _tensor_bytes(output)

        handles = [
            module.register_forward_hook(hook)
            for module in model.modules()
            if module is not model and not any(module.children())
        ]
        output_handle = model.register_forward_hook(
            lambda _module, _inputs, output: captured_outputs.append(output)
        )
        try:
            loss = _candidate_call(
                "abstract",
                adapter.training_step,
                model,
                batch,
                context,
            )
        finally:
            output_handle.remove()
            for handle in handles:
                handle.remove()
        if captured_outputs:
            _validate_output_contract(captured_outputs[-1], manifest)
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise ConfirmedFailure(
                "LOS001",
                "FakeTensor execution did not produce a scalar loss tensor",
                {"loss": _describe(loss)},
            )
    return {
        "backend": "fake_tensor",
        "batch": _describe(batch),
        "loss": _describe(loss),
        "estimated_forward_activation_bytes": activation_bytes,
        "torch_export_attempted": False,
        "torch_export_succeeded": False,
    }


def _abstract_stage(
    manifest: dict[str, Any], scenario: dict[str, Any]
) -> dict[str, Any]:
    try:
        return _abstract_meta_stage(manifest, scenario)
    except CandidateRaised as meta_raised:
        if not _unsupported_abstract(meta_raised):
            raise
        meta_reason = (
            f"{type(meta_raised.original).__name__}: {meta_raised.original}"
        )
    try:
        return _abstract_fake_stage(manifest, scenario)
    except CandidateRaised as fake_raised:
        allow_fallback = manifest["execution"].get(
            "allow_real_cpu_abstract_fallback", False
        )
        if not allow_fallback or not _unsupported_abstract(fake_raised):
            raise
        evidence = _abstract_real_cpu_stage(manifest, scenario)
        evidence["fallback_reason"] = (
            f"meta={meta_reason}; "
            f"fake={type(fake_raised.original).__name__}: {fake_raised.original}"
        )[:1000]
        return evidence


def _failure_diagnostic(
    stage: str,
    failure: ConfirmedFailure,
    scenario: dict[str, Any],
) -> Diagnostic:
    return Diagnostic(
        code=failure.code,
        severity=Severity.ERROR,
        stage=stage,
        classification=Classification.CONFIRMED,
        message=failure.message,
        scenario=scenario,
        evidence=failure.evidence,
        reproduction=f"model-preflight check preflight.yaml --only {stage}",
    )


def _raised_diagnostic(
    stage: str,
    raised: CandidateRaised,
    scenario: dict[str, Any],
) -> Diagnostic:
    exception = raised.original
    message = str(exception)
    lowered = message.lower()
    code = "CHK001"
    classification = Classification.CONFIRMED
    severity = Severity.ERROR
    if isinstance(exception, ModuleNotFoundError):
        code = "SRC002"
    elif isinstance(exception, PermissionError) and "network access is disabled" in lowered:
        code = "NET001"
    elif stage == StageName.ABSTRACT_FORWARD.value:
        patterns = [
            (
                (
                    "mat1 and mat2",
                    "matmul",
                    "shapes cannot be multiplied",
                    "same reduction dim",
                ),
                "SHP002",
            ),
            (("size of tensor", "must match", "broadcast"), "SHP003"),
            (("shape", "invalid for input", "view size"), "SHP004"),
            (("target", "expected long", "scalar type long"), "DAT002"),
            (("target size", "batch_size", "cross_entropy"), "LOS001"),
            (("channel", "channels"), "DAT001"),
        ]
        matched = next(
            (item_code for terms, item_code in patterns if any(term in lowered for term in terms)),
            None,
        )
        if matched is not None:
            code = matched
        else:
            code = "CHK001"
            classification = Classification.INCONCLUSIVE
            severity = Severity.WARNING
    elif raised.area == "import":
        code = "SRC002"
    elif raised.area == "construction":
        code = "CON001"
    elif raised.area == "optimizer":
        code = "OPT001"
    elif raised.area == "fixture":
        code = "FIX001"
        classification = Classification.INCONCLUSIVE
        severity = Severity.WARNING
    elif raised.area in {"training", "backward", "optimizer_step"}:
        if any(term in lowered for term in ("target", "cross_entropy", "binary_cross_entropy")):
            code = "LOS001"
        elif any(term in lowered for term in ("mat1 and mat2", "matmul")):
            code = "SHP002"
        elif any(term in lowered for term in ("size of tensor", "must match")):
            code = "SHP003"
        else:
            code = "AUT002"
    elif raised.area == "validation":
        code = "VAL001"
    return Diagnostic(
        code=code,
        severity=severity,
        stage=stage,
        classification=classification,
        message=f"{raised.area} raised {type(exception).__name__}: {message}",
        scenario=scenario,
        exception_type=type(exception).__name__,
        stack_trace=compact_stacktrace(raised.trace),
        reproduction=f"model-preflight check preflight.yaml --only {stage}",
    )


def _run(request: dict[str, Any]) -> WorkerResponse:
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unsupported protocol version")
    stage = str(request["stage"])
    manifest = dict(request["manifest"])
    scenario = dict(request.get("scenario", {}))
    seed = manifest["execution"].get("seed")
    if seed is not None:
        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            np.random.seed(seed)
    runtime_evidence: dict[str, Any] = {}
    reduction_allowed = bool(request.get("shape_reduction_allowed", False))
    handlers: dict[str, Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]] = {
        StageName.CONSTRUCTION.value: _construct_stage,
        StageName.DATA_CONTRACT.value: _data_stage,
        StageName.CPU_TRAINING.value: _cpu_training_stage,
        StageName.VALIDATION.value: _validation_stage,
        StageName.ABSTRACT_FORWARD.value: _abstract_stage,
    }
    if stage not in handlers:
        raise ValueError(f"unknown worker stage: {stage}")
    try:
        if stage in {StageName.CPU_TRAINING.value, StageName.VALIDATION.value}:
            handler = (
                _cpu_training_stage if stage == StageName.CPU_TRAINING.value else _validation_stage
            )
            evidence = handler(manifest, scenario, runtime_evidence, reduction_allowed)
            evidence.update(runtime_evidence)
        else:
            evidence = handlers[stage](manifest, scenario)
        return WorkerResponse(status=StageStatus.PASS.value, diagnostics=[], evidence=evidence)
    except ConfirmedFailure as failure:
        diagnostic = _failure_diagnostic(stage, failure, scenario)
        return WorkerResponse(
            status=StageStatus.FAIL.value,
            diagnostics=[diagnostic.to_dict()],
            evidence=runtime_evidence,
        )
    except CandidateRaised as raised:
        diagnostic = _raised_diagnostic(stage, raised, scenario)
        status = (
            StageStatus.INCONCLUSIVE
            if diagnostic.classification is Classification.INCONCLUSIVE
            else StageStatus.FAIL
        )
        return WorkerResponse(
            status=status.value,
            diagnostics=[diagnostic.to_dict()],
            evidence=runtime_evidence,
        )


def _capture_protocol() -> tuple[int, Any, Any]:
    protocol_fd = os.dup(1)
    # These files intentionally outlive this helper and are closed at process exit.
    output = tempfile.TemporaryFile(mode="w+", encoding="utf-8")  # noqa: SIM115
    errors = tempfile.TemporaryFile(mode="w+", encoding="utf-8")  # noqa: SIM115
    os.dup2(output.fileno(), 1)
    os.dup2(errors.fileno(), 2)
    sys.stdout = os.fdopen(1, "w", encoding="utf-8", closefd=False)
    sys.stderr = os.fdopen(2, "w", encoding="utf-8", closefd=False)
    return protocol_fd, output, errors


def _captured(stream: Any, limit: int) -> str:
    stream.flush()
    stream.seek(0)
    value = str(stream.read(limit))
    if stream.read(1):
        value += "\n[output truncated]"
    return value


def main() -> None:
    request = json.loads(sys.stdin.read())
    protocol_fd, output, errors = _capture_protocol()
    maximum_output = int(
        request.get("manifest", {})
        .get("execution", {})
        .get("maximum_output_bytes", 1_000_000)
    )
    if os.environ.get("MODEL_PREFLIGHT_DISABLE_NETWORK") == "1":
        sys.addaudithook(_block_network)
    try:
        started = time.monotonic()
        # Apply limits after exec, before importing PyTorch or candidate code. A
        # preexec_fn can deadlock when the controller launches workers from threads.
        if "resource_limits" in request:
            apply_resource_limits(**request["resource_limits"])
        cpu_threads = int(request.get("cpu_threads", 1))
        if cpu_threads > 1:
            # Fail before native libraries can hang trying to create a thread
            # under a UID-wide process limit that is already exhausted.
            probe = Thread()
            try:
                probe.start()
            except RuntimeError as exc:
                raise OSError(
                    errno.EAGAIN,
                    "worker cannot create CPU threads; check maximum_processes and host PID limits",
                ) from exc
            probe.join()
        import torch

        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(1)
        response = _run(request)
        response.evidence["cpu_threads"] = torch.get_num_threads()
    except BaseException as exc:
        resource_failure = isinstance(exc, OSError) and exc.errno in {errno.EAGAIN, errno.ENOMEM}
        diagnostic = Diagnostic(
            code="CHK001" if resource_failure else "CHK003",
            severity=Severity.WARNING if resource_failure else Severity.ERROR,
            stage=str(request.get("stage", "worker")),
            classification=(
                Classification.INCONCLUSIVE if resource_failure else Classification.CHECKER_ERROR
            ),
            message=(
                f"worker resource limit: {type(exc).__name__}: {exc}" if resource_failure
                else f"internal worker failure: {type(exc).__name__}: {exc}"
            ),
            exception_type=type(exc).__name__,
            stack_trace=compact_stacktrace(traceback.format_exc()),
        )
        response = WorkerResponse(
            status=(
                StageStatus.INCONCLUSIVE.value
                if resource_failure else StageStatus.INTERNAL_ERROR.value
            ),
            diagnostics=[diagnostic.to_dict()],
            evidence={"termination_cause": "resource_limit"} if resource_failure else {},
        )
    response.evidence.update({
        "worker_execution_seconds": time.monotonic() - started,
        "peak_worker_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        ),
    })
    response = WorkerResponse(
        status=response.status,
        diagnostics=response.diagnostics,
        evidence=response.evidence,
        captured_stdout=_captured(output, maximum_output // 2),
        captured_stderr=_captured(errors, maximum_output // 2),
    )
    payload = json.dumps(response.to_dict(), sort_keys=True).encode("utf-8")
    os.write(protocol_fd, payload)
    os.close(protocol_fd)


if __name__ == "__main__":
    main()
