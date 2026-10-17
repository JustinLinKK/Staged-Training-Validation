import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from model_preflight.core.enums import StageName, StageStatus
from model_preflight.core.results import StageResult
from model_preflight.engine.cache import cache_key
from model_preflight.engine.runner import PreflightRunner
from model_preflight.execution.shape_reduction import reduce_model
from model_preflight.execution.worker_main import _run
from model_preflight.execution.worker_protocol import WorkerRequest
from model_preflight.manifest import load_manifest
from model_preflight.manifest.models import ShapeReductionConfig
from scripts.shape_reduction_corpus import architecture_cases, audit_cases
from scripts.shape_reduction_corpus import manifest as audit_manifest

SETTINGS = {"mode": "auto", "width_factor": 0.5, "minimum_width": 16}


class Adapter:
    def __init__(self, factory: Any, shape: tuple[int, ...] = (8,)) -> None:
        self.factory = factory
        self.shape = shape

    def build_model(self, context: Any) -> Any:
        return self.factory()

    def build_optimizer(self, model: Any, context: Any) -> Any:
        return torch.optim.SGD(model.parameters(), lr=0.01)

    def build_train_batch(self, scenario: Any, device: str) -> Any:
        return torch.randn(2, *self.shape, device=device), torch.zeros(
            2, dtype=torch.long, device=device
        )

    build_validation_batch = build_train_batch

    def training_step(self, model: Any, batch: Any, context: Any) -> Any:
        return nn.functional.cross_entropy(model(batch[0]), batch[1])

    def validation_step(self, model: Any, batch: Any, context: Any) -> Any:
        return model(batch[0])


def mlp() -> nn.Module:
    return nn.Sequential(nn.Linear(8, 64), nn.LayerNorm(64), nn.ReLU(), nn.Linear(64, 2))


def reduce(adapter: Adapter) -> tuple[Any, dict[str, Any]]:
    model = adapter.build_model({})
    evidence = reduce_model(model, adapter, {"scenario": {}, "device": "cpu"}, SETTINGS)
    return model, evidence


def test_linear_norm_coupling_preserves_values_names_rng_and_external_shape() -> None:
    adapter = Adapter(mlp)
    model = adapter.build_model({})
    original = model[0].weight.detach().clone()
    names = list(dict(model.named_parameters()))
    state = torch.random.get_rng_state().clone()
    evidence = reduce_model(model, adapter, {"scenario": {}}, SETTINGS)
    assert evidence["applied"], evidence
    assert torch.equal(state, torch.random.get_rng_state())
    assert list(dict(model.named_parameters())) == names
    assert model[0].out_features == model[1].normalized_shape[0] == model[3].in_features == 32
    assert model(torch.randn(2, 8)).shape == (2, 2)
    torch.testing.assert_close(model[0].weight, original[:32])
    assert model[0].weight.untyped_storage().nbytes() == 32 * 8 * 4
    loss = adapter.training_step(model, adapter.build_train_batch({}, "cpu"), {})
    loss.backward()
    optimizer = adapter.build_optimizer(model, {})
    assert {id(p) for p in optimizer.param_groups[0]["params"]} == {
        id(p) for p in model.parameters()
    }
    optimizer.step()


@pytest.mark.parametrize("dimension", [1, 2])
def test_conv_norm_pool_flatten_coupling(dimension: int) -> None:
    conv = nn.Conv1d if dimension == 1 else nn.Conv2d
    norm = nn.BatchNorm1d if dimension == 1 else nn.BatchNorm2d
    pool = nn.AdaptiveAvgPool1d if dimension == 1 else nn.AdaptiveAvgPool2d
    adapter = Adapter(
        lambda: nn.Sequential(
            conv(3, 64, 3, padding=1),
            norm(64),
            nn.ReLU(),
            pool(2),
            nn.Flatten(),
            nn.Linear(64 * 2**dimension, 2),
        ),
        (3,) + (8,) * dimension,
    )
    model, evidence = reduce(adapter)
    assert evidence["applied"], evidence
    assert model[0].out_channels == model[1].num_features == 32
    assert model[1].running_mean.shape == (32,)
    assert model[-1].in_features == 32 * 2**dimension
    assert model(torch.randn(2, *adapter.shape)).shape == (2, 2)


class Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(8, 64)
        self.branch = nn.Linear(64, 64)
        self.out = nn.Linear(64, 2)

    def forward(self, value: Any) -> Any:
        value = self.first(value)
        return self.out(value + self.branch(value))


def test_residual_widths_are_unified() -> None:
    model, evidence = reduce(Adapter(Residual))
    assert evidence["applied"], evidence
    assert model.first.out_features == model.branch.in_features == model.branch.out_features == 32
    assert model.out.in_features == 32


class Concat(Residual):
    def __init__(self) -> None:
        super().__init__()
        self.out = nn.Linear(128, 2)

    def forward(self, value: Any) -> Any:
        value = self.first(value)
        return self.out(torch.cat([value, self.branch(value)], dim=1))


def test_concatenation_boundaries_are_pinned() -> None:
    model, evidence = reduce(Adapter(Concat))
    assert not evidence["applied"]
    assert model.first.out_features == 64


class WidthBranch(Residual):
    def forward(self, value: Any) -> Any:
        result = super().forward(value)
        return torch.relu(result) if self.first.out_features > 48 else result


def test_width_dependent_workflow_change_falls_back_without_mutation() -> None:
    model, evidence = reduce(Adapter(WidthBranch))
    assert not evidence["applied"]
    assert evidence["reason"] == "workflow_graph_changed"
    assert model.first.out_features == 64


def test_optimizer_shape_assumptions_fall_back() -> None:
    class WidthOptimizer(Adapter):
        def build_optimizer(self, model: Any, context: Any) -> Any:
            return torch.optim.SGD(model.parameters(), lr=1 / model[0].out_features)

    _, evidence = reduce(WidthOptimizer(mlp))
    assert evidence["reason"] == "optimizer_configuration_changed"


def test_nonfinite_tail_parameter_is_not_sliced_away() -> None:
    adapter = Adapter(mlp)
    model = adapter.build_model({})
    with torch.no_grad():
        model[0].weight[-1, 0] = float("nan")
    evidence = reduce_model(model, adapter, {"scenario": {}}, SETTINGS)
    assert evidence["reason"] == "nonfinite_original_parameters"
    assert model[0].out_features == 64


def test_shared_parameters_and_unsupported_modules_fall_back() -> None:
    def shared() -> nn.Module:
        model = mlp()
        model.register_parameter("alias", model[0].weight)
        return model

    _, evidence = reduce(Adapter(shared))
    assert evidence["reason"] == "shared_modules_or_parameters"
    _, evidence = reduce(
        Adapter(
            lambda: nn.Sequential(
                nn.Conv2d(3, 64, 3),
                nn.Conv2d(64, 64, 1, groups=64),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 2),
            ),
            (3, 8, 8),
        )
    )
    assert evidence["reason"] == "grouped_convolution"


def test_original_width_retry_keeps_both_attempts_and_total_duration() -> None:
    manifest = load_manifest(Path(__file__).parents[1] / "fixtures" / "valid.yaml")
    runner = PreflightRunner(manifest)
    requests = []

    def run(request: Any, timeout: float) -> StageResult:
        requests.append(request)
        return StageResult(
            name="cpu_training",
            duration_seconds=2,
            status=StageStatus.FAIL if request.shape_reduction_allowed else StageStatus.PASS,
            evidence={"shape_reduction": {"applied": request.shape_reduction_allowed}},
        )

    runner.subprocess.run = run  # type: ignore[method-assign]
    result = runner._worker(StageName.CPU_TRAINING, {}, 10, reduction_allowed=True)
    assert result.status is StageStatus.PASS
    assert result.duration_seconds == 4
    assert result.evidence["reduction_attempt"]["status"] == "FAIL"
    assert not result.evidence["shape_reduction"]["applied"]
    assert [r.shape_reduction_allowed for r in requests] == [True, False]


def test_reduction_and_seed_have_separate_cache_entries() -> None:
    manifest = load_manifest(Path(__file__).parents[1] / "fixtures" / "valid.yaml")
    baseline = cache_key(PreflightRunner(manifest)._cache_inputs())
    reduced = replace(
        manifest,
        execution=replace(
            manifest.execution,
            shape_reduction=ShapeReductionConfig(mode="auto"),
        ),
    )
    seeded = replace(manifest, execution=replace(manifest.execution, seed=1))
    assert (
        len(
            {
                baseline,
                cache_key(PreflightRunner(reduced)._cache_inputs()),
                cache_key(PreflightRunner(seeded)._cache_inputs()),
            }
        )
        == 3
    )


def test_seed_reproduces_training_loss() -> None:
    manifest = load_manifest(Path(__file__).parents[1] / "fixtures" / "valid.yaml")
    configured = manifest.to_dict()
    configured["execution"]["seed"] = 11
    request = WorkerRequest(
        stage="cpu_training", manifest=configured,
        scenario=manifest.scenarios.cpu_scenarios()[0],
    ).to_dict()
    first, second = _run(request), _run(request)
    assert first.status == second.status == "PASS"
    assert first.evidence["loss"] == second.evidence["loss"]


@pytest.mark.parametrize(
    "blocked_stage", [None, "construction", "data_contract", "abstract_forward"]
)
def test_prerequisite_gate_and_original_memory(blocked_stage: str | None) -> None:
    manifest = load_manifest(Path(__file__).parents[1] / "fixtures" / "valid.yaml")
    manifest = replace(
        manifest,
        execution=replace(
            manifest.execution,
            shape_reduction=ShapeReductionConfig(mode="auto"),
        ),
    )
    runner = PreflightRunner(manifest)
    requests = []

    def run(request: Any, timeout: float) -> StageResult:
        requests.append(request)
        return StageResult(
            name=request.stage,
            status=StageStatus.INCONCLUSIVE if request.stage == blocked_stage else StageStatus.PASS,
            evidence={
                "parameter_bytes": 1000,
                "trainable_parameter_bytes": 1000,
                "shape_reduction": {"applied": request.shape_reduction_allowed},
            },
        )

    runner.subprocess.run = run  # type: ignore[method-assign]
    report = runner.run()
    runtime = [r for r in requests if r.stage in {"cpu_training", "validation"}]
    assert all(r.shape_reduction_allowed == (blocked_stage is None) for r in runtime)
    assert all(not r.shape_reduction_allowed for r in requests if r not in runtime)
    assert report.gpu_check_required == (blocked_stage is None)
    if blocked_stage != "construction":
        assert report.resource_estimate is not None
        assert report.resource_estimate["parameter_bytes_exact"] == 1000


def test_only_cpu_cannot_enable_reduction() -> None:
    manifest = load_manifest(Path(__file__).parents[1] / "fixtures" / "valid.yaml")
    manifest = replace(
        manifest,
        execution=replace(
            manifest.execution,
            shape_reduction=ShapeReductionConfig(mode="auto"),
        ),
    )
    runner = PreflightRunner(manifest, only={"cpu_training"})
    requests = []

    def run(request: Any, timeout: float) -> StageResult:
        requests.append(request)
        return StageResult(name=request.stage, status=StageStatus.PASS)

    runner.subprocess.run = run  # type: ignore[method-assign]
    runner.run()
    assert requests and all(not r.shape_reduction_allowed for r in requests)


class MultiInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 16))
        self.second = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 16))
        self.head = nn.Linear(32, 2)

    def forward(self, first: Any, second: Any) -> Any:
        return self.head(torch.cat([self.first(first), self.second(second)], dim=1))


def test_nested_batch_and_multiple_model_inputs() -> None:
    class MultiAdapter(Adapter):
        def build_train_batch(self, scenario: Any, device: str) -> Any:
            x, y = super().build_train_batch(scenario, device)
            return (x, x), y

        build_validation_batch = build_train_batch

        def training_step(self, model: Any, batch: Any, context: Any) -> Any:
            return nn.functional.cross_entropy(model(*batch[0]), batch[1])

        def validation_step(self, model: Any, batch: Any, context: Any) -> Any:
            return model(*batch[0])

    model, evidence = reduce(MultiAdapter(MultiInput))
    assert evidence["applied"], evidence
    assert model.first[0].out_features == model.second[0].out_features == 32
    assert model.head.in_features == 32


def test_parameter_storage_aliases_fall_back() -> None:
    adapter = Adapter(mlp)
    model = adapter.build_model({})
    model.register_parameter("view_alias", nn.Parameter(model[0].weight.view_as(model[0].weight)))
    evidence = reduce_model(model, adapter, {"scenario": {}}, SETTINGS)
    assert evidence["reason"] == "shared_tensor_storage"


def test_validation_submodule_output_contract_cannot_shrink() -> None:
    class HiddenValidation(Adapter):
        def validation_step(self, model: Any, batch: Any, context: Any) -> Any:
            return model[0](batch[0])

    model, evidence = reduce(HiddenValidation(mlp))
    assert not evidence["applied"]
    assert evidence["reason"] == "workflow_output_contract_changed"
    assert model[0].out_features == 64


def test_cached_model_cannot_be_used_as_disposable_probe() -> None:
    cached = mlp()
    adapter = Adapter(lambda: cached)
    original = {name: value.detach().clone() for name, value in cached.state_dict().items()}
    evidence = reduce_model(cached, adapter, {"scenario": {}}, SETTINGS)
    assert evidence["reason"] == "non_disposable_meta_model"
    assert cached[0].out_features == 64
    for name, value in cached.state_dict().items():
        torch.testing.assert_close(value, original[name])


@pytest.mark.parametrize("operation", ["method", "function", "exp2", "power", "division", "sqrt"])
def test_sensitive_loss_math_retains_original_width(operation: str) -> None:
    class SensitiveLoss(Adapter):
        def training_step(self, model: Any, batch: Any, context: Any) -> Any:
            value = model(batch[0])
            if operation == "method":
                return value.exp().mean()
            if operation == "function":
                return torch.exp(value).mean()
            if operation == "exp2":
                return torch.exp2(value).mean()
            if operation == "power":
                return (value ** 2).mean()
            if operation == "division":
                return (value / value).mean()
            return value.sqrt().mean()

    model, evidence = reduce(SensitiveLoss(mlp))
    assert not evidence["applied"]
    assert evidence["reason"].startswith("numerically_sensitive_operation:")
    assert model[0].out_features == 64


def test_sensitive_validation_math_retains_original_width() -> None:
    class SensitiveValidation(Adapter):
        def validation_step(self, model: Any, batch: Any, context: Any) -> Any:
            return torch.exp(model(batch[0]))

    model, evidence = reduce(SensitiveValidation(mlp))
    assert evidence["reason"] == "numerically_sensitive_operation:exp"
    assert model[0].out_features == 64


@pytest.mark.parametrize(
    "case", audit_cases() + architecture_cases(), ids=lambda case: case["name"]
)
def test_reduction_audit_worker_outcomes(tmp_path: Path, case: dict[str, Any]) -> None:
    for dependency in case.get("requires", []):
        pytest.importorskip(dependency)
    if case["expected_status"] is None:
        pytest.skip(case["label_note"])
    (tmp_path / f"{case['name']}.py").write_text(case["source"])
    configured = audit_manifest(case, tmp_path)
    configured["candidate"]["adapter"] = f"{case['name']}:CandidateAdapter"
    configured["execution"].update({"shape_reduction": SETTINGS, "seed": 11})
    config_path = tmp_path / "preflight.json"
    config_path.write_text(json.dumps(configured))
    manifest = load_manifest(config_path)
    responses = [
        _run(WorkerRequest(
            stage=stage, manifest=manifest.to_dict(),
            scenario=manifest.scenarios.cpu_scenarios()[0], shape_reduction_allowed=True,
        ).to_dict())
        for stage in ("cpu_training", "validation")
    ]
    codes = {d["code"] for response in responses for d in response.diagnostics}
    if case["expected_status"] == "FAIL":
        assert any(response.status == "FAIL" for response in responses), responses
        assert set(case["expected_codes"]).issubset(codes), codes
    else:
        assert all(response.status == "PASS" for response in responses), responses
    if case.get("expect_reduction"):
        assert all(response.evidence["shape_reduction"]["applied"] for response in responses)
    if case.get("expect_fallback"):
        assert all(not response.evidence["shape_reduction"]["applied"] for response in responses)


def test_probe_context_mutation_cannot_change_the_real_scenario() -> None:
    class StatefulContext(Adapter):
        def training_step(self, model: Any, batch: Any, context: Any) -> Any:
            context["scenario"]["calls"] = context["scenario"].get("calls", 0) + 1
            return super().training_step(model, batch, context)

    adapter = StatefulContext(mlp)
    model = adapter.build_model({})
    context = {"scenario": {}, "device": "cpu"}
    evidence = reduce_model(model, adapter, context, SETTINGS)
    assert evidence["reason"] == "mutable_workflow_context"
    assert context == {"scenario": {}, "device": "cpu"}
    assert model[0].out_features == 64


class TransformerModel(nn.Module):
    def __init__(self, family: str, norm_first: bool = False) -> None:
        super().__init__()
        if family == "encoder":
            self.block = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                32, 4, 128, dropout=0, batch_first=True, norm_first=norm_first,
            ), 2, enable_nested_tensor=False)
        elif family == "decoder":
            self.block = nn.TransformerDecoder(nn.TransformerDecoderLayer(
                32, 4, 128, dropout=0, batch_first=True, norm_first=norm_first,
            ), 2)
        else:
            self.block = nn.Transformer(32, 4, 2, 2, 128, dropout=0, batch_first=True,
                                        norm_first=norm_first)
        self.family = family
        self.head = nn.Linear(32, 2)

    def forward(self, x: Any) -> Any:
        result = self.block(x) if self.family == "encoder" else self.block(x, x)
        return self.head(result.mean(dim=1))


@pytest.mark.parametrize("family", ["encoder", "decoder", "encoder_decoder"])
@pytest.mark.parametrize("norm_first", [False, True])
def test_transformer_shrinks_only_feedforward(family: str, norm_first: bool) -> None:
    adapter = Adapter(lambda: TransformerModel(family, norm_first), (4, 32))
    model = adapter.build_model({})
    parameters = dict(model.named_parameters())
    preserved = {name: p.detach().clone() for name, p in parameters.items()
                 if ".linear1." not in name and ".linear2.weight" not in name}
    evidence = reduce_model(model, adapter, {"scenario": {}}, SETTINGS)
    assert evidence["applied"], evidence
    assert evidence["reduced_parameter_count"] < evidence["original_parameter_count"]
    assert parameters.keys() == dict(model.named_parameters()).keys()
    for module in model.modules():
        if type(module) in {nn.TransformerEncoderLayer, nn.TransformerDecoderLayer}:
            assert module.linear1.out_features == module.linear2.in_features == 64
            assert module.self_attn.embed_dim == 32 and module.self_attn.num_heads == 4
    for name, tensor in preserved.items():
        torch.testing.assert_close(model.get_parameter(name), tensor)
    optimizer = adapter.build_optimizer(model, {})
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in model.parameters()
    }
    for _ in range(2):
        optimizer.zero_grad()
        adapter.training_step(model, adapter.build_train_batch({}, "cpu"), {}).backward()
        assert all(p.grad is not None for p in model.parameters())
        optimizer.step()
    model.eval()
    with torch.no_grad():
        assert model(torch.randn(2, 4, 32)).shape == (2, 2)


@pytest.mark.parametrize("unsupported", ["layer", "activation", "method"])
def test_transformer_unknown_internals_fall_back_atomically(unsupported: str) -> None:
    def factory() -> Any:
        model = TransformerModel("encoder")
        if unsupported == "layer":
            model.block.layers[1].linear2 = nn.Sequential(nn.Linear(128, 32), nn.Identity())
        elif unsupported == "activation":
            model.block.layers[1].activation = torch.sigmoid
        else:
            model.block.layers[1]._ff_block = lambda x: x.detach()
        return model

    adapter = Adapter(factory, (4, 32))
    model = adapter.build_model({})
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    evidence = reduce_model(model, adapter, {"scenario": {}}, SETTINGS)
    assert not evidence["applied"], evidence
    assert model.block.layers[0].linear1.out_features == 128
    for name, tensor in original.items():
        torch.testing.assert_close(model.get_parameter(name), tensor)


def mamba_model() -> Any:
    transformers = pytest.importorskip("transformers")
    module = pytest.importorskip("transformers.models.mamba.modeling_mamba")
    config = transformers.MambaConfig(hidden_size=32, state_size=8, expand=2,
                                      use_associative_scan=False, use_mambapy=False)
    return nn.Sequential(module.MambaMixer(config, 0), nn.Flatten(), nn.Linear(4 * 32, 2))


def test_real_mamba_mixer_preserves_gate_channel_alignment_and_state_axes() -> None:
    adapter = Adapter(mamba_model, (4, 32))
    model = adapter.build_model({})
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    state = torch.random.get_rng_state().clone()
    evidence = reduce_model(model, adapter, {"scenario": {}}, SETTINGS)
    assert evidence["applied"], evidence
    assert torch.equal(state, torch.random.get_rng_state())
    assert original.keys() == dict(model.named_parameters()).keys()
    mixer = model[0]
    assert mixer.intermediate_size == mixer.conv1d.groups == 32
    assert (mixer.hidden_size, mixer.ssm_state_size, mixer.time_step_rank) == (32, 8, 2)
    torch.testing.assert_close(mixer.in_proj.weight,
                              torch.cat((original["0.in_proj.weight"][:32],
                                         original["0.in_proj.weight"][64:96])))
    torch.testing.assert_close(mixer.A_log, original["0.A_log"][:32])
    torch.testing.assert_close(mixer.x_proj.weight, original["0.x_proj.weight"][:, :32])
    optimizer = adapter.build_optimizer(model, {})
    for _ in range(2):
        optimizer.zero_grad()
        adapter.training_step(model, adapter.build_train_batch({}, "cpu"), {}).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    model.eval()
    assert model(torch.randn(2, 4, 32)).shape == (2, 2)


def test_mamba_transition_overflow_cannot_be_sliced_away() -> None:
    adapter = Adapter(mamba_model, (4, 32))
    model = adapter.build_model({})
    with torch.no_grad():
        model[0].A_log[-1].fill_(100)
    evidence = reduce_model(model, adapter, {"scenario": {}}, SETTINGS)
    assert evidence["reason"] == "nonfinite_original_mamba_transition"
    assert model[0].intermediate_size == 64


@pytest.mark.parametrize("family", ["transformer", "mamba"])
def test_structured_layer_can_be_the_root_model(family: str) -> None:
    class SequenceAdapter(Adapter):
        def training_step(self, model: Any, batch: Any, context: Any) -> Any:
            return model(batch[0]).sum()

    factory = (lambda: nn.TransformerEncoderLayer(32, 4, 128, batch_first=True)) if (
        family == "transformer"
    ) else lambda: mamba_model()[0]
    model, evidence = reduce(SequenceAdapter(factory, (4, 32)))
    assert evidence["applied"], evidence
    assert model(torch.randn(2, 4, 32)).shape == (2, 4, 32)


def test_unknown_layer_after_transformer_cancels_the_entire_plan() -> None:
    adapter = Adapter(lambda: nn.Sequential(
        nn.TransformerEncoderLayer(32, 4, 128, batch_first=True),
        nn.Softmax(dim=-1), nn.Flatten(), nn.Linear(128, 2),
    ), (4, 32))
    model = adapter.build_model({})
    parameters = {name: id(p) for name, p in model.named_parameters()}
    evidence = reduce_model(model, adapter, {"scenario": {}}, SETTINGS)
    assert evidence["reason"] == "unsupported_module:Softmax"
    assert model[0].linear1.out_features == 128
    assert parameters == {name: id(p) for name, p in model.named_parameters()}


def test_reusing_mamba_projection_outside_mixer_cannot_overwrite_packed_slices() -> None:
    class ReusedProjection(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mixer = mamba_model()[0]
            self.aux = nn.Linear(128, 32)
            self.head = nn.Linear(32, 2)

        def forward(self, x: Any) -> Any:
            hidden = self.mixer(x)
            aux = self.aux(self.mixer.in_proj(x))
            return self.head((hidden + aux).mean(dim=1))

    adapter = Adapter(ReusedProjection, (4, 32))
    model = adapter.build_model({})
    original = model.mixer.in_proj.weight.detach().clone()
    evidence = reduce_model(model, adapter, {"scenario": {}}, SETTINGS)
    assert evidence.get("reason") == "reused_structured_internal", evidence
    assert not evidence["applied"]
    torch.testing.assert_close(model.mixer.in_proj.weight, original)
