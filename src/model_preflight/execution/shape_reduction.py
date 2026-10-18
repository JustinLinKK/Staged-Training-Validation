"""Conservative worker-only width reduction; candidate methods remain authoritative."""

from __future__ import annotations

import copy
import inspect
import math
import operator
import random
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, cast

import torch
from torch import nn
from torch.fx import GraphModule, Node, Tracer
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import functional as F

from model_preflight.execution.worker_protocol import SHAPE_REDUCER_REVISION


class UnsupportedReduction(Exception):
    """The original model must be used for this scenario."""


_TRANSFORMERS = {
    nn.Transformer, nn.TransformerEncoder, nn.TransformerDecoder,
    nn.TransformerEncoderLayer, nn.TransformerDecoderLayer,
}


def _mamba_kind(module: Any) -> str | None:
    kind = type(module)
    namespace = inspect.getmodule(kind)
    if (
        kind.__module__ == "transformers.models.mamba.modeling_mamba"
        and kind.__name__ in {"MambaMixer", "MambaBlock", "MambaModel"}
        and namespace is not None
        and getattr(namespace, kind.__name__, None) is kind
    ):
        return str(kind.__name__)
    return None


def _check_mamba_call(module: Any, args: Any, kwargs: Any) -> None:
    bound = inspect.signature(module.forward).bind(*args, **kwargs)
    if bound.arguments.get("cache_params") is not None:
        raise UnsupportedReduction("mamba_inference_cache")
    if _mamba_kind(module) == "MambaModel":
        use_cache = bound.arguments.get("use_cache")
        if use_cache is None:
            use_cache = module.config.use_cache and not module.training
        if use_cache is not False:
            raise UnsupportedReduction("mamba_inference_cache")


class _ReductionTracer(Tracer):
    def is_leaf_module(self, module: nn.Module, qualified_name: str) -> bool:
        return _mamba_kind(module) is not None or super().is_leaf_module(module, qualified_name)


def _trace(model: Any, concrete_args: Any = None) -> GraphModule:
    graph = GraphModule(model, _ReductionTracer().trace(model, concrete_args=concrete_args))
    structured = [
        str(node.target) for node in graph.graph.nodes if node.op == "call_module"
        and (type(graph.get_submodule(str(node.target))) in _TRANSFORMERS
             or _mamba_kind(graph.get_submodule(str(node.target))) is not None)
    ]
    # A block's packed slices cannot also participate in an independent width plan,
    # including direct parameter access or extra calls from the training adapter.
    if any(
        str(node.target).startswith(path + ".")
        for node in graph.graph.nodes if node.op in {"call_module", "get_attr"}
        for path in structured
    ):
        raise UnsupportedReduction("reused_structured_internal")
    return graph


def _structured_plan(module: Any, settings: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Only resize known internal expansions; residual, attention and state axes stay fixed."""
    def width(original: int) -> int:
        return min(original, max(int(settings["minimum_width"]),
                                 int(original * settings["width_factor"]) // 8 * 8))

    for child in module.modules():
        if any(callable(value) for key, value in vars(child).items() if key != "activation"):
            raise UnsupportedReduction("custom_structured_method")
    mamba_kind = _mamba_kind(module)
    if mamba_kind in {"MambaModel", "MambaBlock"}:
        result = {}
        namespace = inspect.getmodule(type(module))
        for name, child in module.named_modules():
            if getattr(child, "gradient_checkpointing", False):
                raise UnsupportedReduction("mamba_gradient_checkpointing")
            if _mamba_kind(child) == "MambaMixer":
                for key, values in _structured_plan(child, settings).items():
                    result[name + ("." + key if key else "")] = values
            elif _mamba_kind(child) in {"MambaModel", "MambaBlock"}:
                expected = ({"embeddings", "layers", "norm_f"}
                            if _mamba_kind(child) == "MambaModel" else {"norm", "mixer"})
                if set(dict(child.named_children())) != expected:
                    raise UnsupportedReduction("unsupported_mamba_structure")
            elif (type(child) not in {nn.ModuleList, nn.Embedding}
                  and type(child) is not getattr(namespace, "MambaRMSNorm", None)
                  and not any(name.startswith(path + ".") for path, m in module.named_modules()
                              if _mamba_kind(m) == "MambaMixer")):
                raise UnsupportedReduction(f"unsupported_mamba_module:{type(child).__name__}")
        return result
    if mamba_kind == "MambaMixer":
        if module.use_mambapy or getattr(module, "use_associative_scan", False):
            raise UnsupportedReduction("unsupported_mamba_scan_backend")
        children = dict(module.named_children())
        expected = {"conv1d", "act", "in_proj", "x_proj", "dt_proj", "out_proj"}
        if set(children) != expected or any(
            type(children[name]) is not nn.Linear
            for name in ("in_proj", "x_proj", "dt_proj", "out_proj")
        ) or type(module.conv1d) is not nn.Conv1d or (
            type(module.act).__module__, type(module.act).__name__
        ) != ("transformers.activations", "SiLUActivation"):
            raise UnsupportedReduction("unsupported_mamba_structure")
        inner, hidden = module.intermediate_size, module.hidden_size
        state, rank = module.ssm_state_size, module.time_step_rank
        if (
            module.in_proj.weight.shape != (2 * inner, hidden)
            or module.x_proj.weight.shape != (rank + 2 * state, inner)
            or module.dt_proj.weight.shape != (inner, rank)
            or module.out_proj.weight.shape != (hidden, inner)
            or module.conv1d.weight.shape != (inner, 1, module.conv_kernel_size)
            or module.conv1d.groups != inner
            or module.A_log.shape != (inner, state) or module.D.shape != (inner,)
        ):
            raise UnsupportedReduction("unsupported_mamba_dimensions")
        reduced = width(inner)
        return {"": {"intermediate_size": reduced}} if reduced < inner else {}
    allowed = _TRANSFORMERS | {
        nn.ModuleList, nn.Linear, nn.LayerNorm, nn.Dropout, nn.MultiheadAttention,
        nn.modules.linear.NonDynamicallyQuantizableLinear,
    }
    result = {}
    for name, child in module.named_modules():
        if type(child) not in allowed:
            raise UnsupportedReduction(f"unsupported_transformer_module:{type(child).__name__}")
        if type(child) not in {nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}:
            continue
        if child.activation not in (F.relu, F.gelu):
            raise UnsupportedReduction("unsupported_transformer_activation")
        if type(child.linear1) is not nn.Linear or type(child.linear2) is not nn.Linear:
            raise UnsupportedReduction("unsupported_transformer_feedforward")
        inner = child.linear1.out_features
        hidden = child.self_attn.embed_dim
        if (child.linear1.in_features != hidden or child.linear2.in_features != inner
                or child.linear2.out_features != hidden):
            raise UnsupportedReduction("unsupported_transformer_dimensions")
        reduced = width(inner)
        if reduced < inner:
            prefix = name + "." if name else ""
            result[prefix + "linear1"] = {"in_features": hidden, "out_features": reduced}
            result[prefix + "linear2"] = {"in_features": reduced, "out_features": hidden}
    return result


@contextmanager
def _preserve_rng() -> Iterator[None]:
    python_state = random.getstate()
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore[assignment]
    numpy_state = np.random.get_state() if np is not None else None
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        random.setstate(python_state)
        if np is not None and numpy_state is not None:
            np.random.set_state(numpy_state)


def _model_signature(model: Any) -> Any:
    return (
        [
            (
                name,
                type(module),
                {
                    key: value
                    for key, value in vars(module).items()
                    if not key.startswith("_")
                    and isinstance(value, (str, bool, int, float, type(None)))
                },
            )
            for name, module in model.named_modules()
        ],
        [(name, tuple(p.shape), p.dtype, p.requires_grad) for name, p in model.named_parameters()],
        [(name, tuple(b.shape), b.dtype) for name, b in model.named_buffers()],
    )


def _graph_signature(graph: GraphModule) -> Any:
    return [(n.op, n.target, str(n.args), str(n.kwargs)) for n in graph.graph.nodes]


def _contract(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), str(value.dtype), value.requires_grad)
    if isinstance(value, dict):
        return {key: _contract(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, [_contract(item) for item in value])
    return type(value).__name__


class _Workflow(nn.Module):
    def __init__(self, model: Any, adapter: Any, context: Any, validation: bool) -> None:
        super().__init__()
        self.model = model
        self.adapter = adapter
        self.context = context
        self.validation = validation

    def forward(self, batch: Any) -> Any:
        step = self.adapter.validation_step if self.validation else self.adapter.training_step
        return step(self.model, batch, self.context)


def _workflow_signature(
    model: Any,
    adapter: Any,
    context: Any,
    validation: bool,
    batch: Any,
) -> Any:
    model, adapter = copy.deepcopy((model, adapter))
    if not validation:
        adapter.build_optimizer(model, context)
    builder = adapter.build_validation_batch if validation else adapter.build_train_batch
    batch = builder(context["scenario"], "meta")
    model.train(not validation)
    placeholder: Any = torch.fx.PH  # type: ignore[attr-defined]
    structure = torch.fx.node.map_aggregate(
        batch,
        lambda value: placeholder if isinstance(value, torch.Tensor) else value,
    )
    # Shape/graph equivalence cannot establish numerical equivalence around overflow
    # or singularities. Keep these workflows at original width, including their loss.
    sensitive = {
        "exp", "exp2", "expm1", "log", "log2", "log10", "log1p", "sqrt", "rsqrt",
        "reciprocal", "pow", "float_power", "div", "divide", "true_divide", "truediv",
        "__pow__", "__truediv__",
    }
    steps = 1 if validation else (
        2 if context.get("manifest", {}).get("scenarios", {}).get("stateful_two_steps") else 1
    )
    signatures = []
    for _ in range(steps):
        with torch.set_grad_enabled(not validation):
            graph = _trace(
                _Workflow(model, adapter, context, validation),
                concrete_args={"batch": structure},
            )
        for node in graph.graph.nodes:
            if node.op == "call_module":
                module = graph.get_submodule(str(node.target))
                if _mamba_kind(module) is not None:
                    _check_mamba_call(module, node.args, node.kwargs)
            name = (
                str(node.target) if node.op == "call_method"
                else getattr(node.target, "__name__", "") if node.op == "call_function" else ""
            )
            if name.rstrip("_") in sensitive or name in sensitive:
                raise UnsupportedReduction(f"numerically_sensitive_operation:{name}")
        signatures.append(_graph_signature(graph))
    return signatures


def _optimizer_signature(model: Any, adapter: Any, context: Any) -> Any:
    model, adapter = copy.deepcopy((model, adapter))
    optimizer = adapter.build_optimizer(model, context)
    if not type(optimizer).__module__.startswith("torch.optim.") or "step" in vars(optimizer):
        raise UnsupportedReduction("custom_optimizer")
    if optimizer.state:
        raise UnsupportedReduction("preloaded_optimizer_state")
    if optimizer._optimizer_step_pre_hooks or optimizer._optimizer_step_post_hooks:
        raise UnsupportedReduction("custom_optimizer_hooks")
    names = {id(p): name for name, p in model.named_parameters()}
    return (
        type(optimizer),
        [
            (
                tuple(names[id(p)] for p in group["params"]),
                repr(sorted((k, v) for k, v in group.items() if k != "params")),
            )
            for group in optimizer.param_groups
        ],
    )


def _check_numeric_range(value: Any, origin: str) -> Any:
    if isinstance(value, torch.Tensor) and value.numel():
        if not bool(torch.isfinite(value).all()):
            raise UnsupportedReduction(f"nonfinite_original_{origin}")
        if value.is_floating_point():
            lower, upper = torch.aminmax(value.detach())
            # Extreme finite values can overflow even ordinary linear accumulation.
            # This is a conservative fallback threshold, not a numerical safety proof.
            limit = math.sqrt(torch.finfo(value.dtype).max)
            if float(lower) < -limit or float(upper) > limit:
                raise UnsupportedReduction(f"extreme_original_{origin}")
    return value


def _shape(node: Node) -> tuple[int, ...]:
    metadata = node.meta.get("tensor_meta")
    if metadata is None or not hasattr(metadata, "shape"):
        raise UnsupportedReduction("non_tensor_model_graph")
    return tuple(int(value) for value in metadata.shape)


def _plan(graph: GraphModule, settings: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Unify channel/feature widths, pin boundaries, and propagate flatten factors."""
    parents: list[int] = []
    widths: list[int] = []
    pinned: set[int] = set()
    # Each value is (width group, feature axis, flatten multiplier).
    features: dict[Node, tuple[int, int, int]] = {}
    dimensions: dict[str, dict[str, tuple[int, int]]] = {}
    structured: dict[str, dict[str, int]] = {}

    def group(width: int, fixed: bool = False) -> int:
        index = len(parents)
        parents.append(index)
        widths.append(width)
        if fixed:
            pinned.add(index)
        return index

    def root(index: int) -> int:
        while parents[index] != index:
            index = parents[index]
        return index

    def pin(node: Node) -> None:
        if node in features:
            pinned.add(features[node][0])

    def input_feature(node: Node, axis: int, width: int) -> tuple[int, int, int]:
        if node not in features:
            features[node] = (group(width, True), axis, 1)
        feature = features[node]
        if feature[1] != axis or widths[feature[0]] * feature[2] != width:
            raise UnsupportedReduction("uncertain_feature_axis")
        return feature

    pointwise = {
        nn.ReLU,
        nn.ReLU6,
        nn.GELU,
        nn.SiLU,
        nn.Tanh,
        nn.Sigmoid,
        nn.LeakyReLU,
        nn.ELU,
        nn.Identity,
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
    }
    pools = {
        nn.MaxPool1d,
        nn.MaxPool2d,
        nn.AvgPool1d,
        nn.AvgPool2d,
        nn.AdaptiveAvgPool1d,
        nn.AdaptiveAvgPool2d,
    }
    functions = {
        torch.relu,
        torch.sigmoid,
        torch.tanh,
        F.relu,
        F.relu6,
        F.gelu,
        F.silu,
        F.leaky_relu,
        F.elu,
        F.dropout,
        F.dropout1d,
        F.dropout2d,
    }
    for node in graph.graph.nodes:
        if node.op in {"placeholder", "get_attr"}:
            continue
        if node.op == "output":
            torch.fx.map_arg(node.args, lambda n: pin(n))
            continue
        if node.op == "call_module":
            module: Any = graph.get_submodule(str(node.target))
            if type(module) in _TRANSFORMERS or _mamba_kind(module) is not None:
                if _mamba_kind(module) is not None:
                    _check_mamba_call(module, node.args, node.kwargs)
                torch.fx.map_arg((node.args, node.kwargs), lambda n: pin(n))
                for name, values in _structured_plan(module, settings).items():
                    path = str(node.target) + ("." + name if name else "")
                    if path in structured:
                        raise UnsupportedReduction("reused_parameterized_module")
                    structured[path] = values
                continue
            if type(module) in {nn.MultiheadAttention, nn.Embedding}:
                # Opaque fixed-width boundaries; never change Q/K/V, heads or vocabulary.
                torch.fx.map_arg((node.args, node.kwargs), lambda n: pin(n))
                continue
            source = node.args[0]
            if not isinstance(source, Node):
                raise UnsupportedReduction("non_tensor_module_input")
            shape = _shape(source)
            kind = type(module)
            if str(node.target) in dimensions:
                raise UnsupportedReduction("reused_parameterized_module")
            if kind in {nn.Linear, nn.Conv1d, nn.Conv2d}:
                if kind is nn.Linear:
                    incoming, outgoing, axis = (
                        module.in_features,
                        module.out_features,
                        len(shape) - 1,
                    )
                    in_name, out_name = "in_features", "out_features"
                else:
                    if module.groups != 1:
                        raise UnsupportedReduction("grouped_convolution")
                    incoming, outgoing, axis = module.in_channels, module.out_channels, 1
                    in_name, out_name = "in_channels", "out_channels"
                previous = input_feature(source, axis, incoming)
                next_group = group(outgoing)
                features[node] = (next_group, axis, 1)
                dimensions[str(node.target)] = {
                    in_name: (previous[0], previous[2]),
                    out_name: (next_group, 1),
                }
            elif kind in {nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm}:
                axis = len(shape) - 1 if kind is nn.LayerNorm else 1
                if kind is nn.LayerNorm and len(module.normalized_shape) != 1:
                    raise UnsupportedReduction("multiaxis_layer_norm")
                width = module.normalized_shape[0] if kind is nn.LayerNorm else module.num_features
                feature = input_feature(source, axis, width)
                features[node] = feature
                name = "normalized_shape" if kind is nn.LayerNorm else "num_features"
                dimensions[str(node.target)] = {name: (feature[0], feature[2])}
            elif kind in pointwise | pools:
                if source in features:
                    features[node] = features[source]
            elif kind is nn.Flatten:
                _flatten(source, node, features)
            else:
                raise UnsupportedReduction(f"unsupported_module:{kind.__name__}")
        elif node.op == "call_function" and node.target in functions:
            source = node.args[0]
            if source in features:
                features[node] = features[source]
        elif node.op == "call_function" and node.target in {operator.add, torch.add}:
            left, right = node.args[:2]
            if not isinstance(left, Node) or not isinstance(right, Node):
                raise UnsupportedReduction("scalar_or_unknown_residual")
            if left not in features or right not in features:
                pin(left)
                pin(right)
            else:
                a, b = features[left], features[right]
                if _shape(left) != _shape(right) or a[1:] != b[1:] or widths[a[0]] != widths[b[0]]:
                    raise UnsupportedReduction("broadcast_residual")
                parents[root(b[0])] = root(a[0])
                features[node] = a
        elif node.op == "call_function" and node.target is torch.cat:
            torch.fx.map_arg(node.args, lambda n: pin(n))
            # Concatenated widths are fixed; closed regions on either side may still shrink.
        elif (node.op == "call_function" and node.target in {
            operator.getitem, torch.mean, torch.sum,
        }) or (node.op == "call_method" and node.target in {"mean", "sum"}):
            # Selection and pooling are boundaries, including attention's tuple result.
            torch.fx.map_arg((node.args, node.kwargs), lambda n: pin(n))
        elif (node.op == "call_function" and node.target is torch.flatten) or (
            node.op == "call_method" and node.target == "flatten"
        ):
            _flatten(node.args[0], node, features)
        elif node.op == "call_method" and node.target in {"squeeze", "unsqueeze", "contiguous"}:
            source = node.args[0]
            if source in features:
                feature = features[source]
                before, after = _shape(source), _shape(node)
                axes = [i for i, value in enumerate(after) if value == before[feature[1]]]
                if node.target == "contiguous":
                    features[node] = feature
                elif len(axes) == 1 and before[feature[1]] != 1:
                    features[node] = (feature[0], axes[0], feature[2])
                else:
                    pin(source)
        else:
            raise UnsupportedReduction(f"unsupported_operation:{node.op}:{node.target}")
    fixed = {root(index) for index in pinned}
    selected = {
        index: (
            widths[index]
            if root(index) in fixed
            else min(
                widths[index],
                max(
                    int(settings["minimum_width"]),
                    int(widths[index] * settings["width_factor"]) // 8 * 8,
                ),
            )
        )
        for index in range(len(widths))
    }
    return structured | {
        name: {key: selected[index] * multiplier for key, (index, multiplier) in values.items()}
        for name, values in dimensions.items()
        if any(selected[index] != widths[index] for index, _ in values.values())
    }


def _flatten(source: Any, node: Node, features: dict[Node, tuple[int, int, int]]) -> None:
    if source not in features:
        return
    group, axis, multiplier = features[source]
    before, after = _shape(source), _shape(node)
    if axis != 1 or len(after) != 2 or after[0] != before[0]:
        raise UnsupportedReduction("uncertain_flatten")
    features[node] = (group, 1, multiplier * math.prod(before[2:]))


def _apply(model: Any, plan: dict[str, dict[str, int]]) -> None:
    # Allocate every replacement before mutating the model; never retain full-sized views.
    changes: list[tuple[Any, str, Any]] = []
    for name, dimensions in plan.items():
        module = model.get_submodule(name)
        if _mamba_kind(module) == "MambaMixer":
            width = dimensions["intermediate_size"]
            original = module.intermediate_size
            # in_proj packs [signal, gate]; each half must retain corresponding channels.
            indices = torch.cat((torch.arange(width, device=module.in_proj.weight.device),
                                 torch.arange(original, original + width,
                                              device=module.in_proj.weight.device)))
            slices = [
                (module.in_proj, "weight", module.in_proj.weight[indices]),
                (module.x_proj, "weight", module.x_proj.weight[:, :width]),
                (module.dt_proj, "weight", module.dt_proj.weight[:width]),
                (module.dt_proj, "bias", module.dt_proj.bias[:width]),
                (module.out_proj, "weight", module.out_proj.weight[:, :width]),
                (module.conv1d, "weight", module.conv1d.weight[:width]),
                (module, "A_log", module.A_log[:width]), (module, "D", module.D[:width]),
            ]
            if module.in_proj.bias is not None:
                slices.append((module.in_proj, "bias", module.in_proj.bias[indices]))
            if module.conv1d.bias is not None:
                slices.append((module.conv1d, "bias", module.conv1d.bias[:width]))
            for owner, key, sliced in slices:
                original_parameter = getattr(owner, key)
                value = nn.Parameter(sliced.detach().clone(),
                                     requires_grad=original_parameter.requires_grad)
                value.__dict__.update(copy.deepcopy(vars(original_parameter)))
                changes.append((owner, key, value))
            for owner, key, size in (
                (module.in_proj, "out_features", width * 2),
                (module.x_proj, "in_features", width),
                (module.dt_proj, "out_features", width),
                (module.out_proj, "in_features", width),
                (module.conv1d, "in_channels", width),
                (module.conv1d, "out_channels", width),
                (module.conv1d, "groups", width),
            ):
                changes.append((owner, key, size))
        elif type(module) in {nn.Linear, nn.Conv1d, nn.Conv2d}:
            output = dimensions.get("out_features", dimensions.get("out_channels"))
            incoming = dimensions.get("in_features", dimensions.get("in_channels"))
            changes.append(
                (
                    module,
                    "weight",
                    nn.Parameter(
                        module.weight[:output, :incoming].detach().clone(),
                        requires_grad=module.weight.requires_grad,
                    ),
                )
            )
            if module.bias is not None:
                changes.append(
                    (
                        module,
                        "bias",
                        nn.Parameter(
                            module.bias[:output].detach().clone(),
                            requires_grad=module.bias.requires_grad,
                        ),
                    )
                )
        else:
            width = next(iter(dimensions.values()))
            for key in ("weight", "bias", "running_mean", "running_var"):
                tensor = getattr(module, key, None)
                if tensor is not None:
                    value = tensor[:width].detach().clone()
                    if isinstance(tensor, nn.Parameter):
                        value = nn.Parameter(value, requires_grad=tensor.requires_grad)
                    changes.append((module, key, value))
        for key, size in dimensions.items():
            changes.append((module, key, (size,) if key == "normalized_shape" else size))
    for module, key, value in changes:
        setattr(module, key, value)


def reduce_model(
    model: Any,
    adapter: Any,
    context: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Attempt reduction atomically; all unsupported paths retain the original instance."""
    started = time.monotonic()
    original_count = sum(p.numel() for p in model.parameters())
    evidence: dict[str, Any] = {
        "applied": False,
        "execution_model": "original",
        "reducer_revision": SHAPE_REDUCER_REVISION,
        "original_parameter_count": original_count,
        "reduced_parameter_count": original_count,
    }
    try:
        modules = list(model.named_modules(remove_duplicate=False))
        parameters = list(model.named_parameters(remove_duplicate=False))
        if len({id(m) for _, m in modules}) != len(modules) or len(
            {id(p) for _, p in parameters}
        ) != len(parameters):
            raise UnsupportedReduction("shared_modules_or_parameters")
        tensors = [p for _, p in parameters] + list(model.buffers())
        storage = [t.untyped_storage().data_ptr() for t in tensors if t.numel()]
        if len(set(storage)) != len(storage):
            raise UnsupportedReduction("shared_tensor_storage")
        if any(p._backward_hooks or p._post_accumulate_grad_hooks for _, p in parameters):
            raise UnsupportedReduction("custom_parameter_hooks")
        for method in (adapter.training_step, adapter.validation_step):
            try:
                source = inspect.getsource(method)
            except (OSError, TypeError):
                raise UnsupportedReduction("unavailable_workflow_source") from None
            if re.search(r"\.apply\s*\(", source):
                raise UnsupportedReduction("custom_autograd")
        for _, module in modules:
            if type(module).__module__.startswith(("mamba_ssm.", "transformers.models.mamba2.")):
                raise UnsupportedReduction(f"unsupported_mamba_backend:{type(module).__name__}")
            if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks:
                raise UnsupportedReduction("custom_hooks")
            namespace = inspect.getmodule(type(module))
            if not type(module).__module__.startswith("torch."):
                try:
                    source = inspect.getsource(type(module).forward)
                except (OSError, TypeError):
                    raise UnsupportedReduction("unavailable_forward_source") from None
                if re.search(r"\.apply\s*\(", source):
                    raise UnsupportedReduction("custom_autograd")
            if (
                namespace is not None
                and not type(module).__module__.startswith("torch.")
                and any(
                    isinstance(v, type) and issubclass(v, torch.autograd.Function)
                    for v in vars(namespace).values()
                )
            ):
                raise UnsupportedReduction("custom_autograd")
        if not any(
            type(m) in {nn.Linear, nn.Conv1d, nn.Conv2d}
            and max(m.weight.shape[:2]) > settings["minimum_width"]
            for _, m in modules
        ):
            raise UnsupportedReduction("no_reducible_width")
        for _, parameter in parameters:
            _check_numeric_range(parameter, "parameters")
        for buffer in model.buffers():
            _check_numeric_range(buffer, "buffers")
        for _, module in modules:
            if _mamba_kind(module) == "MambaMixer":
                _structured_plan(module, settings)
                if not bool(torch.isfinite(torch.exp(module.A_log.detach().float())).all()):
                    raise UnsupportedReduction("nonfinite_original_mamba_transition")
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)) and (
                module.running_var is not None and bool((module.running_var < 0).any())
            ):
                raise UnsupportedReduction("invalid_normalization_variance")
        with _preserve_rng():
            batch_adapter = copy.deepcopy(adapter)
            for builder in (batch_adapter.build_train_batch, batch_adapter.build_validation_batch):
                torch.fx.node.map_aggregate(
                    builder(copy.deepcopy(context["scenario"]), "cpu"),
                    lambda value: _check_numeric_range(value, "batch"),
                )
        with _preserve_rng(), torch.device("meta"):
            probe_adapter = copy.deepcopy(adapter)
            probe_context = copy.deepcopy({**context, "device": "meta"})
            initial_context = copy.deepcopy(probe_context)
            probe = probe_adapter.build_model(probe_context)
            original_modules = {id(module) for _, module in modules}
            if any(id(module) in original_modules for module in probe.modules()) or any(
                tensor.device.type != "meta" for tensor in [*probe.parameters(), *probe.buffers()]
            ):
                raise UnsupportedReduction("non_disposable_meta_model")
            if _model_signature(probe) != _model_signature(model):
                raise UnsupportedReduction("cpu_meta_model_mismatch")
            before_optimizer = _optimizer_signature(probe, probe_adapter, probe_context)
            input_probe, input_adapter = copy.deepcopy((probe, probe_adapter))
            input_adapter.build_optimizer(input_probe, probe_context)
            batch = input_adapter.build_train_batch(probe_context["scenario"], "meta")
            input_probe.train()
            graph_probe = copy.deepcopy(input_probe)
            inputs: list[tuple[Any, Any]] = []
            handle = input_probe.register_forward_pre_hook(
                lambda _m, args, kwargs: inputs.append((args, kwargs)),
                with_kwargs=True,
            )
            try:
                original_train = _contract(
                    input_adapter.training_step(input_probe, batch, probe_context)
                )
            finally:
                handle.remove()
            if len(inputs) != 1:
                raise UnsupportedReduction("requires_one_model_call")
            if type(graph_probe) in _TRANSFORMERS or _mamba_kind(graph_probe) is not None:
                plan = _structured_plan(graph_probe, settings)
                if _mamba_kind(graph_probe) is not None:
                    _check_mamba_call(graph_probe, inputs[0][0], inputs[0][1])
            else:
                graph = _trace(graph_probe)
                bound = inspect.signature(graph.forward).bind(*inputs[0][0], **inputs[0][1])
                bound.apply_defaults()
                cast(Any, ShapeProp)(graph).propagate(*bound.arguments.values())
                plan = _plan(graph, settings)
            if not plan:
                raise UnsupportedReduction("no_reducible_hidden_dimensions")
            validation_batch = copy.deepcopy(probe_adapter).build_validation_batch(
                probe_context["scenario"], "meta"
            )
            before_train = _workflow_signature(probe, probe_adapter, probe_context, False, batch)
            before_eval = _workflow_signature(
                probe,
                probe_adapter,
                probe_context,
                True,
                validation_batch,
            )
            validation_probe, validation_adapter = copy.deepcopy((probe, probe_adapter))
            validation_batch = validation_adapter.build_validation_batch(
                probe_context["scenario"], "meta"
            )
            validation_probe.eval()
            with torch.no_grad():
                original_validation = _contract(
                    validation_adapter.validation_step(
                        validation_probe, validation_batch, probe_context
                    )
                )
            required = getattr(probe_adapter, "required_gradient_names", None)
            before_required = set(required(probe)) if callable(required) else None
            original_dimensions = {
                name: {key: getattr(probe.get_submodule(name), key) for key in values}
                for name, values in plan.items()
            }
            _apply(probe, plan)
            if before_train != _workflow_signature(
                probe, probe_adapter, probe_context, False, batch
            ) or before_eval != _workflow_signature(
                probe,
                probe_adapter,
                probe_context,
                True,
                validation_batch,
            ):
                raise UnsupportedReduction("workflow_graph_changed")
            if before_optimizer != _optimizer_signature(probe, probe_adapter, probe_context):
                raise UnsupportedReduction("optimizer_configuration_changed")
            if callable(required) and before_required != set(required(probe)):
                raise UnsupportedReduction("required_gradients_changed")
            input_probe, input_adapter = copy.deepcopy((probe, probe_adapter))
            input_adapter.build_optimizer(input_probe, probe_context)
            batch = input_adapter.build_train_batch(probe_context["scenario"], "meta")
            input_probe.train()
            reduced_train = _contract(
                input_adapter.training_step(input_probe, batch, probe_context)
            )
            validation_probe, validation_adapter = copy.deepcopy((probe, probe_adapter))
            validation_batch = validation_adapter.build_validation_batch(
                probe_context["scenario"], "meta"
            )
            validation_probe.eval()
            with torch.no_grad():
                reduced_validation = _contract(
                    validation_adapter.validation_step(
                        validation_probe, validation_batch, probe_context
                    )
                )
            if original_train != reduced_train or original_validation != reduced_validation:
                raise UnsupportedReduction("workflow_output_contract_changed")
            if probe_context != initial_context:
                raise UnsupportedReduction("mutable_workflow_context")
        _apply(model, plan)
        evidence.update(
            {
                "applied": True,
                "execution_model": "reduced",
                "reduced_parameter_count": sum(p.numel() for p in model.parameters()),
                "dimensions": {
                    name: {"original": original_dimensions[name], "reduced": values}
                    for name, values in plan.items()
                },
            }
        )
    except UnsupportedReduction as exc:
        evidence["reason"] = str(exc)
    except Exception as exc:
        evidence["reason"] = f"reduction_unavailable:{type(exc).__name__}: {exc}"[:500]
    evidence["duration_seconds"] = time.monotonic() - started
    return evidence
