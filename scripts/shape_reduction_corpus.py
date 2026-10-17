"""Fixed synthetic comparison cases. Labels are independent of checker outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tests.mutations.cases import MUTATIONS

HEADER = """
import torch
from torch import nn
from torch.nn import functional as F
"""

ADAPTER = """
class CandidateAdapter:
    def build_model(self, context):
        return Model()
    def build_optimizer(self, model, context):
        return torch.optim.SGD(model.parameters(), lr=0.001)
    def build_train_batch(self, scenario, device):
        n = scenario["batch_size"]
        inputs = torch.randn(n, *SHAPE, device=device)
        return inputs, torch.zeros(n, dtype=torch.long, device=device)
    def build_validation_batch(self, scenario, device):
        return self.build_train_batch(scenario, device)
    def training_step(self, model, batch, context):
        return F.cross_entropy(model(batch[0]), batch[1])
    def validation_step(self, model, batch, context):
        return model(batch[0])
"""

MLP = """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(128, WIDTH), nn.LayerNorm(WIDTH), nn.GELU(),
                                    nn.Linear(WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, 2))
    def forward(self, x):
        return self.layers(x)
"""
CNN = """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, WIDTH, 3, padding=1), nn.BatchNorm2d(WIDTH),
                                    nn.ReLU(), nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(WIDTH, 2))
    def forward(self, x):
        return self.layers(x)
"""
RESIDUAL = """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, WIDTH, 3, padding=1)
        self.branch = nn.Sequential(nn.Conv2d(WIDTH, WIDTH, 3, padding=1), nn.BatchNorm2d(WIDTH),
                                    nn.ReLU(), nn.Conv2d(WIDTH, WIDTH, 3, padding=1))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(WIDTH, 2))
    def forward(self, x):
        x = self.stem(x)
        return self.head(F.relu(x + self.branch(x)))
"""
MULTIMODAL = """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(nn.Conv2d(3, WIDTH, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(WIDTH, 32, 3, padding=1), nn.ReLU(),
                                   nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.tabular = nn.Sequential(nn.Linear(12, WIDTH), nn.ReLU(), nn.Linear(WIDTH, 16))
        self.head = nn.Sequential(nn.Linear(48, WIDTH), nn.ReLU(), nn.Linear(WIDTH, 2))
    def forward(self, image, tabular):
        return self.head(torch.cat([self.image(image), self.tabular(tabular)], dim=1))
"""


def manifest(case: dict[str, Any], root: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "candidate": {
            "id": case["name"],
            "root": str(root),
            "adapter": "candidate:CandidateAdapter",
        },
        "task": {
            "name": "classification",
            "num_classes": 2,
            "target_dtype": "int64",
            "output_class_dimension": -1,
        },
        "scenarios": {
            "train_batch_sizes": [case.get("batch_size", 4)],
            "last_batch_size": 1,
            "test_last_batch": True,
            "run_validation": True,
            "input_shapes": {"normal": case.get("shape", [128])},
            "precision": ["fp32"],
            "stateful_two_steps": case.get("two_steps", True),
        },
        "target": {"profile": "nvidia/a10_24gb"},
        "execution": {
            "abstract_timeout_seconds": 30,
            "cpu_timeout_seconds": 60,
            "maximum_cpu_memory_mb": 8192,
            "cache": False,
        },
        "policy": {"mode": "balanced"},
    }


def cases() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for mutation in MUTATIONS:
        result.append(
            {
                "name": "existing_" + mutation.name,
                "source": mutation.source,
                "group": "existing",
                "expected_status": "FAIL",
                "expected_codes": [mutation.code],
                "shape": [4],
            }
        )
    for family, model, widths, shape, batch in (
        ("mlp", MLP, (512, 2048), [128], 64),
        ("cnn", CNN, (64, 128), [3, 64, 64], 8),
        ("residual", RESIDUAL, (64, 128), [3, 64, 64], 8),
        ("multimodal", MULTIMODAL, (64, 128), [3, 64, 64], 8),
    ):
        for width in widths:
            adapter = ADAPTER
            if family == "multimodal":
                adapter = adapter.replace(
                    "inputs = torch.randn(n, *SHAPE, device=device)",
                    "inputs = (torch.randn(n, *SHAPE, device=device), "
                    "torch.randn(n, 12, device=device))",
                ).replace("model(batch[0])", "model(*batch[0])")
            source = HEADER + f"\nWIDTH = {width}\nSHAPE = {shape!r}\n" + model + adapter
            result.append(
                {
                    "name": f"{family}_{width}",
                    "source": source,
                    "group": "performance",
                    "expected_status": "PASS",
                    "expected_codes": [],
                    "performance": True,
                    "shape": shape,
                    "batch_size": batch,
                    "expect_reduction": True,
                }
            )
    base = HEADER + "\nWIDTH = 64\nSHAPE = [128]\n" + MLP + ADAPTER
    mutations = [
        ("detached_loss", "AUT001", "return F.cross_entropy(model(batch[0]), batch[1]).detach()"),
        (
            "vector_loss",
            "LOS001",
            "return F.cross_entropy(model(batch[0]), batch[1], reduction='none')",
        ),
        ("nonfinite_loss", "NUM001", "return model(batch[0]).sum() * float('nan')"),
        (
            "nonfinite_gradient",
            "NUM002",
            (
                "return model(batch[0]).sum() * 0 + "
                "(model.layers[0].bias[0] - model.layers[0].bias[0].detach()).sqrt()"
            ),
        ),
        (
            "second_backward",
            "AUT002",
            'if not hasattr(self, "saved"):\n'
            "            self.saved = F.cross_entropy(model(batch[0]), batch[1])\n"
            "        return self.saved",
        ),
        (
            "width_dependent_detach",
            "AUT001",
            "loss = F.cross_entropy(model(batch[0]), batch[1])\n"
            "        return loss.detach() if model.layers[0].out_features > 48 else loss",
        ),
    ]
    for name, code, loss in mutations:
        result.append(
            {
                "name": name,
                "source": base.replace("return F.cross_entropy(model(batch[0]), batch[1])", loss),
                "group": "adversarial" if name.startswith("width_") else "workflow",
                "expected_status": "FAIL",
                "expected_codes": [code],
            }
        )
    result.append(
        {
            "name": "missing_gradient",
            "source": base.replace(
                "self.layers = nn.Sequential",
                "self.unused = nn.Linear(128, 2)\n        self.layers = nn.Sequential",
            )
            + '\n    def required_gradient_names(self, model):\n        return {"unused.weight"}\n',
            "group": "workflow",
            "expected_status": "FAIL",
            "expected_codes": ["AUT003"],
        }
    )
    result.append(
        {
            "name": "validation_failure",
            "source": base.replace(
                "return model(batch[0])", 'raise RuntimeError("validation failure")'
            ),
            "group": "workflow",
            "expected_status": "FAIL",
            "expected_codes": ["VAL001"],
        }
    )
    optimizer = """
class BrokenSGD(torch.optim.SGD):
    def step(self, closure=None):
        raise RuntimeError("optimizer update failed")
"""
    result.append(
        {
            "name": "optimizer_step",
            "source": base.replace(
                "class CandidateAdapter:", optimizer + "\nclass CandidateAdapter:"
            ).replace("torch.optim.SGD(model.parameters()", "BrokenSGD(model.parameters()"),
            "group": "workflow",
            "expected_status": "FAIL",
            "expected_codes": ["AUT002"],
        }
    )
    result.append(
        {
            "name": "optimizer_shape_assumption",
            "source": base.replace("lr=0.001", "lr=1 / model.layers[0].out_features"),
            "group": "adversarial",
            "expected_status": "PASS",
            "expected_codes": [],
        }
    )
    result.append(
        {
            "name": "tail_channel_nan",
            "source": base.replace(
                "return Model()",
                "model = Model()\n        with torch.no_grad():\n"
                '            model.layers[0].weight[-1, 0] = float("nan")\n'
                "        return model",
            ),
            "group": "adversarial",
            "expected_status": "FAIL",
            "expected_codes": ["NUM001"],
        }
    )
    result.append(
        {
            "name": "reshape_constant",
            "source": base.replace(
                "return self.layers(x)",
                "return self.layers[5](self.layers[0](x).reshape(x.shape[0], 64))",
            ),
            "group": "adversarial",
            "expected_status": "PASS",
            "expected_codes": [],
        }
    )
    result.append(
        {
            "name": "shared_parameters",
            "source": base.replace(
                "return Model()",
                "model = Model()\n"
                '        model.register_parameter("alias", model.layers[0].weight)\n'
                "        return model",
            ),
            "group": "adversarial",
            "expected_status": "PASS",
            "expected_codes": [],
        }
    )
    overflow = """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(128, 128)
        self.last = nn.Linear(128, 2)
        nn.init.zeros_(self.first.weight)
        nn.init.ones_(self.first.bias)
        nn.init.ones_(self.last.weight)
        nn.init.zeros_(self.last.bias)
    def forward(self, x):
        return self.last(self.first(x))
"""
    result.append(
        {
            "name": "width_dependent_overflow",
            "source": HEADER
            + "\nSHAPE = [128]\n"
            + overflow
            + ADAPTER.replace("lr=0.001", "lr=0.0").replace(
                "F.cross_entropy(model(batch[0]), batch[1])", "model(batch[0]).exp().mean()"
            ),
            "group": "adversarial",
            "expected_status": "FAIL",
            "expected_codes": ["NUM001"],
            "two_steps": False,
        }
    )
    for name, model, shape in [
        (
            "grouped_conv",
            CNN.replace(
                "nn.Conv2d(WIDTH, WIDTH, 3, padding=1)",
                "nn.Conv2d(WIDTH, WIDTH, 3, padding=1, groups=WIDTH)",
            ),
            [3, 8, 8],
        ),
        (
            "dynamic_control",
            MLP.replace(
                "return self.layers(x)", "return self.layers(x) if x.sum() > 0 else self.layers(-x)"
            ),
            [128],
        ),
        (
            "recurrent",
            """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(128, 64, batch_first=True)
        self.head = nn.Linear(64, 2)
    def forward(self, x):
        return self.head(self.rnn(x)[0][:, -1])
""",
            [4, 128],
        ),
        (
            "attention",
            """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.MultiheadAttention(64, 4, batch_first=True)
        self.head = nn.Linear(64, 2)
    def forward(self, x):
        return self.head(self.attention(x, x, x, need_weights=False)[0][:, 0])
""",
            [4, 64],
        ),
    ]:
        result.append(
            {
                "name": name,
                "source": HEADER + f"\nWIDTH = 64\nSHAPE = {shape!r}\n" + model + ADAPTER,
                "group": "unsupported",
                "expected_status": "INCONCLUSIVE" if name == "dynamic_control" else "PASS",
                "expected_codes": [],
                "shape": shape,
            }
        )
    result.append(
        {
            "name": "cached_model",
            "group": "unsupported",
            "source": HEADER
            + "\nWIDTH = 64\nSHAPE = [128]\n"
            + MLP
            + "\nCACHED_MODEL = Model()\n"
            + ADAPTER.replace("return Model()", "return CACHED_MODEL"),
            "expected_status": "PASS",
            "expected_codes": [],
        }
    )
    result.extend(audit_cases())
    result.extend(architecture_cases())
    return result


def architecture_cases() -> list[dict[str, Any]]:
    """Real library layers, with architecture-specific reduction and fallback controls."""
    result: list[dict[str, Any]] = []

    def add(name: str, model: str, code: str | None = None, **settings: Any) -> None:
        result.append({
            "name": "architecture_" + name, "source": HEADER + "\nSHAPE = [4, 32]\n"
            + model + ADAPTER, "group": "architecture", "shape": [4, 32],
            "expected_status": "FAIL" if code else "PASS",
            "expected_codes": [code] if code else [], **settings,
        })

    template = """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = BLOCK
        self.head = nn.Linear(32, 2)
    def forward(self, x):
        result = CALL
        return self.head(result.mean(dim=1))
"""
    encoder = template.replace("BLOCK", "nn.TransformerEncoder(nn.TransformerEncoderLayer("
                               "32, 4, 128, dropout=0.1, batch_first=True), 2, "
                               "enable_nested_tensor=False)").replace("CALL", "self.block(x)")
    decoder = template.replace("BLOCK", "nn.TransformerDecoder(nn.TransformerDecoderLayer("
                               "32, 4, 128, dropout=0.1, batch_first=True), 2)").replace(
                                   "CALL", "self.block(x, x)")
    seq2seq = template.replace("BLOCK", "nn.Transformer(32, 4, 2, 2, 128, dropout=0.1, "
                               "batch_first=True)").replace("CALL", "self.block(x, x)")
    for name, source in (("transformer_encoder", encoder), ("transformer_decoder", decoder),
                         ("transformer_seq2seq", seq2seq)):
        add(name, source, expect_reduction=True)
    add("transformer_mask", encoder.replace(
        "self.head =", "self.register_buffer('mask', torch.ones(4, 4, dtype=torch.bool).triu(1))"
        "\n        self.head =",
    ).replace("self.block(x)", "self.block(x, mask=self.mask, is_causal=True)"),
        expect_reduction=True)
    add("transformer_unknown_layer", "class CustomLinear(nn.Linear):\n    pass\n" + encoder.replace(
        "self.head =", "self.block.layers[1].linear2 = CustomLinear(128, 32)"
        "\n        self.head =",
    ), expect_fallback=True)
    add("transformer_custom_activation", encoder.replace(
        "self.head =", "self.block.layers[1].activation = torch.sigmoid\n        self.head =",
    ), expect_fallback=True)
    add("transformer_width_branch", encoder.replace(
        "return self.head(result.mean(dim=1))",
        "logits = self.head(result.mean(dim=1))\n"
        "        return logits.detach() if self.block.layers[0].linear1.out_features > 96 "
        "else logits",
    ), "AUT001", expect_fallback=True)
    add("transformer_tail_nan", encoder.replace(
        "self.head =", "with torch.no_grad():\n"
        "            self.block.layers[0].linear1.weight[-1].fill_(float('nan'))\n"
        "        self.head =",
    ), "NUM001", expect_fallback=True)
    attention = template.replace("BLOCK", "nn.MultiheadAttention(32, 4, batch_first=True)")
    attention = attention.replace("CALL", "self.block(x, x, x, need_weights=False)[0]")
    attention = attention.replace("nn.Linear(32, 2)",
                                  "nn.Sequential(nn.Linear(32, 128), nn.GELU(), nn.Linear(128, 2))")
    add("attention_mlp", attention, expect_reduction=True)
    mamba = """
from transformers import MambaConfig
from transformers.models.mamba.modeling_mamba import MambaMixer
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        config = MambaConfig(hidden_size=32, state_size=8, expand=2,
                             use_associative_scan=False, use_mambapy=False)
        self.mixer = MambaMixer(config, 0)
        self.head = nn.Linear(32, 2)
    def forward(self, x):
        return self.head(self.mixer(x).mean(dim=1))
"""
    add("mamba_mixer", mamba, expect_reduction=True, requires=["transformers"])
    add("mamba_stack", mamba.replace("self.head =", "self.second = MambaMixer(config, 1)\n"
                                    "        self.head =").replace(
                                        "self.mixer(x)", "self.second(self.mixer(x))"),
        expect_reduction=True, requires=["transformers"])
    add("mamba_reused_projection", mamba.replace(
        "self.head =", "self.aux = nn.Linear(128, 32)\n        self.head =",
    ).replace("self.mixer(x).mean(dim=1)",
              "(self.mixer(x) + self.aux(self.mixer.in_proj(x))).mean(dim=1)"),
        expect_fallback=True, requires=["transformers"])
    add("mamba_width_branch", mamba.replace(
        "return self.head(self.mixer(x).mean(dim=1))",
        "logits = self.head(self.mixer(x).mean(dim=1))\n"
        "        return logits.detach() if self.mixer.intermediate_size > 48 else logits",
    ), "AUT001", expect_fallback=True, requires=["transformers"])
    add("mamba_tail_nan", mamba.replace(
        "self.head =", "with torch.no_grad():\n"
        "            self.mixer.D[-1].fill_(float('nan'))\n        self.head =",
    ), "NUM001", expect_fallback=True, requires=["transformers"])
    add("mamba_transition_overflow", mamba.replace(
        "self.head =", "with torch.no_grad():\n"
        "            self.mixer.A_log[-1].fill_(100)\n        self.head =",
    ), "AUT002", expect_fallback=True, requires=["transformers"])
    add("mamba_scan_backend", mamba.replace("use_associative_scan=False",
                                            "use_associative_scan=True"),
        expect_fallback=True, requires=["transformers"])
    language = """
from transformers import MambaConfig, MambaModel
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        config = MambaConfig(vocab_size=32, hidden_size=32, state_size=8, expand=2,
                             num_hidden_layers=2, use_associative_scan=False, use_mambapy=False)
        self.backbone = MambaModel(config)
        self.head = nn.Linear(32, 2)
    def forward(self, x):
        hidden = self.backbone(input_ids=x, use_cache=False, return_dict=False)[0]
        return self.head(hidden.mean(dim=1))
"""
    for name, source, reduced in (
        ("mamba_model", language, True),
        ("mamba_cache", language.replace("use_cache=False, ", ""), False),
    ):
        add(name, source, expect_reduction=reduced, expect_fallback=not reduced,
            requires=["transformers"])
        result[-1]["shape"] = [4]
        result[-1]["source"] = result[-1]["source"].replace(
            "SHAPE = [4, 32]", "SHAPE = [4]"
        ).replace("torch.randn(n, *SHAPE, device=device)",
                  "torch.randint(0, 32, (n, *SHAPE), device=device)")
    mamba2 = mamba.replace("MambaConfig", "Mamba2Config").replace(
        "models.mamba.modeling_mamba import MambaMixer",
        "models.mamba2.modeling_mamba2 import Mamba2Mixer",
    ).replace("MambaMixer(config", "Mamba2Mixer(config").replace(
        "use_associative_scan=False, use_mambapy=False",
        "num_heads=4, head_dim=16, n_groups=1, chunk_size=4",
    )
    add("mamba2_fallback", mamba2, expect_fallback=True, requires=["transformers"])
    native = template.replace("BLOCK", "Mamba(d_model=32, d_state=8, expand=2)").replace(
        "CALL", "self.block(x)"
    )
    add("mamba_native", "from mamba_ssm import Mamba\n" + native,
        expect_fallback=True, requires=["mamba_ssm"])
    result[-1]["expected_status"] = None
    result[-1]["label_note"] = "CUDA-only dependency; CPU outcome requires environment validation"
    return result


def audit_cases() -> list[dict[str, Any]]:
    """Reduction blind spots, labeled from explicit defects before running the checker."""
    base = HEADER + "\nWIDTH = 64\nSHAPE = [128]\n" + MLP + ADAPTER
    result: list[dict[str, Any]] = []

    def add(name: str, source: str, code: str | None, **settings: Any) -> None:
        result.append({
            "name": "audit_" + name, "source": source, "group": "reduction_audit",
            "expected_status": "FAIL" if code else "PASS",
            "expected_codes": [code] if code else [], **settings,
        })

    for step in (1, 2):
        optimizer = f"""
class WidthSGD(torch.optim.SGD):
    def step(self, closure=None):
        self.calls = getattr(self, 'calls', 0) + 1
        if self.calls == {step} and self.param_groups[0]['params'][0].shape[0] > 48:
            raise RuntimeError('width-dependent optimizer failure')
        return super().step(closure)
"""
        add(f"optimizer_step_{step}", base.replace(
            "class CandidateAdapter:", optimizer + "\nclass CandidateAdapter:"
        ).replace("torch.optim.SGD(model.parameters()", "WidthSGD(model.parameters()"), "AUT002")

    add("optimizer_state", base.replace(
        "return torch.optim.SGD(model.parameters(), lr=0.001)",
        "optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)\n"
        "        parameter = model.layers[0].weight\n"
        "        state = torch.zeros_like(parameter)\n"
        "        if parameter.shape[0] > 48:\n"
        "            state[-1] = float('nan')\n"
        "        optimizer.state[parameter]['momentum_buffer'] = state\n"
        "        return optimizer",
    ), "NUM001")

    for step in (1, 2):
        for defect, expression, code in (
            ("detach", "loss.detach()", "AUT001"),
            ("nan", "loss * float('nan')", "NUM001"),
        ):
            add(f"adapter_step_{step}_{defect}", base.replace(
                "return F.cross_entropy(model(batch[0]), batch[1])",
                "self.calls = getattr(self, 'calls', 0) + 1\n"
                "        loss = F.cross_entropy(model(batch[0]), batch[1])\n"
                f"        if self.calls == {step} and model.layers[0].out_features > 48:\n"
                f"            return {expression}\n"
                "        return loss",
            ), code)

    add("model_first_call_detach", base.replace(
        "return self.layers(x)",
        "self.calls = getattr(self, 'calls', 0) + 1\n"
        "        output = self.layers(x)\n"
        "        if self.calls == 1 and self.layers[0].out_features > 48:\n"
        "            return output.detach()\n"
        "        return output",
    ), "AUT001")

    for name, hook, code in (
        ("parameter_hook", "g * float('inf')", "NUM002"),
        ("parameter_hook_control", "g", None),
    ):
        add(name, base.replace(
            "return Model()",
            "model = Model()\n"
            f"        model.layers[0].weight.register_hook(lambda g: {hook})\n"
            "        return model",
        ), code)

    add("adapter_autograd", base.replace(
        "return F.cross_entropy(model(batch[0]), batch[1])",
        "class WidthBackward(torch.autograd.Function):\n"
        "            @staticmethod\n"
        "            def forward(ctx, loss, width):\n"
        "                ctx.width = width\n"
        "                return loss\n"
        "            @staticmethod\n"
        "            def backward(ctx, gradient):\n"
        "                if ctx.width > 48:\n"
        "                    raise RuntimeError('width-dependent backward failure')\n"
        "                return gradient, None\n"
        "        loss = F.cross_entropy(model(batch[0]), batch[1])\n"
        "        return WidthBackward.apply(loss, model.layers[0].out_features)",
    ), "AUT002")

    for name, buffer, value in (
        ("buffer_nan", "running_mean", "float('nan')"),
        ("negative_variance", "running_var", "-1.0"),
    ):
        source = HEADER + "\nWIDTH = 64\nSHAPE = [3, 8, 8]\n" + CNN + ADAPTER
        add(name, source.replace(
            "return Model()", "model = Model()\n"
            f"        model.layers[1].{buffer}[-1] = {value}\n"
            "        return model",
        ), "NUM001", shape=[3, 8, 8])

    add("nested_validation_nan", base.replace(
        "return model(batch[0])",
        "return {'logits': model(batch[0]), 'metrics': [torch.tensor(float('nan'))]}",
    ), "NUM001")
    add("nested_validation_control", base.replace(
        "return model(batch[0])",
        "return {'logits': model(batch[0]), 'metrics': [torch.tensor(0.0)]}",
    ), None, expect_reduction=True)

    add("last_update_nan", base.replace(
        "class CandidateAdapter:", """
class CorruptSGD(torch.optim.SGD):
    def step(self, closure=None):
        result = super().step(closure)
        self.calls = getattr(self, 'calls', 0) + 1
        if self.calls == 2:
            with torch.no_grad():
                self.param_groups[0]['params'][0][-1] = float('nan')
        return result

class CandidateAdapter:"""
    ).replace("torch.optim.SGD(model.parameters()", "CorruptSGD(model.parameters()"), "NUM001")

    linear = """
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(128, 64), nn.Linear(64, 2))
        with torch.no_grad():
            self.layers[0].weight.zero_()
            self.layers[0].bias.fill_(1.0)
            self.layers[0].bias[-1] = 1e30
            self.layers[1].weight.zero_()
            self.layers[1].weight[0, -1] = 1e30
            self.layers[1].bias.zero_()
    def forward(self, x):
        return self.layers(x)
"""
    add("finite_tail_overflow", HEADER + "\nSHAPE = [128]\n" + linear + ADAPTER, "NUM001")
    add("finite_input_overflow", HEADER + "\nSHAPE = [128]\n" + linear.replace(
        "self.layers[0].weight.zero_()", "self.layers[0].weight.fill_(1 / 128)"
    ).replace("self.layers[0].bias.fill_(1.0)", "self.layers[0].bias.zero_()").replace(
        "self.layers[0].bias[-1] = 1e30", "self.layers[0].bias[-1] = 0"
    ).replace("self.layers[1].weight.zero_()", "self.layers[1].weight.fill_(1.0)").replace(
        "self.layers[1].weight[0, -1] = 1e30", "self.layers[1].weight[0, -1] = 1.0"
    ) + ADAPTER.replace(
        "torch.randn(n, *SHAPE, device=device)",
        "torch.full((n, *SHAPE), 1e37, device=device)",
    ), "NUM001")
    add("ordinary_mlp_control", base, None, expect_reduction=True)
    add("context_first_call_nan", base.replace(
        "return F.cross_entropy(model(batch[0]), batch[1])",
        "context['calls'] = context.get('calls', 0) + 1\n"
        "        loss = F.cross_entropy(model(batch[0]), batch[1])\n"
        "        if context['calls'] == 1 and model.layers[0].out_features > 48:\n"
        "            return loss * float('nan')\n"
        "        return loss",
    ), "NUM001")
    add("optimizer_training_mode", base.replace(
        "return Model()", "return Model().eval()"
    ).replace(
        "return torch.optim.SGD(model.parameters(), lr=0.001)",
        "model.optimizer_saw_eval = not model.training\n"
        "        return torch.optim.SGD(model.parameters(), lr=0.001)",
    ).replace(
        "return F.cross_entropy(model(batch[0]), batch[1])",
        "loss = F.cross_entropy(model(batch[0]), batch[1])\n"
        "        if getattr(model, 'optimizer_saw_eval', False) "
        "and model.layers[0].out_features > 48:\n"
        "            return loss.detach()\n"
        "        return loss",
    ), "AUT001")
    return result


def scale_cases() -> list[dict[str, Any]]:
    """Width-only follow-up, retaining one original anchor per architecture family."""
    original = {case["name"]: case for case in cases() if case.get("performance")}
    result = []
    for family, anchor, widths in (
        ("mlp", 2048, (2048, 8192, 16384)),
        ("cnn", 128, (128, 256, 512)),
        ("residual", 128, (128, 256, 512)),
        ("multimodal", 128, (128, 256, 512)),
    ):
        template = original[f"{family}_{anchor}"]
        for width in widths:
            result.append({
                **template,
                "name": f"{family}_{width}",
                "source": template["source"].replace(f"WIDTH = {anchor}\n", f"WIDTH = {width}\n"),
                "family": family,
                "width": width,
                "anchor_width": anchor,
            })
    return result


def archive_cases(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Use source snapshots and immutable reports; summaries are not ground truth."""
    found: list[dict[str, Any]] = []
    exclusions: list[dict[str, str]] = []
    seen: set[str] = set()
    if not root.is_dir():
        return [], [{"source": str(root), "reason": "archive_directory_missing"}]
    import hashlib

    for source in sorted(root.glob("*/*.py")):
        if source.name not in {"draft.py", "injected.py", "repaired.py"}:
            continue
        content = source.read_text()
        digest = hashlib.sha256(content.encode()).hexdigest()
        if digest in seen:
            exclusions.append({"source": str(source), "reason": "duplicate_source"})
            continue
        seen.add(digest)
        # Original checker tree hash for the single-file candidate snapshot.
        tree_hash = "sha256:" + hashlib.sha256(b"candidate.py" + content.encode()).hexdigest()
        reports = []
        for path in sorted((source.parent / "cpu_reports").glob("*.json")):
            report = json.loads(path.read_text())
            if report.get("candidate_hash") == tree_hash:
                reports.append((path, report))
        if not reports:
            exclusions.append({"source": str(source), "reason": "no_matching_immutable_report"})
            continue
        path, report = reports[0]
        original_manifest = report["normalized_manifest"]
        found.append(
            {
                "name": "archive_" + source.parent.name + "_" + source.stem,
                "source": content,
                "source_path": str(source),
                "source_sha256": digest,
                "evidence_path": str(path),
                "group": "historical",
                "expected_status": None,
                "expected_codes": [],
                "label": "historical_observation_not_independent_ground_truth",
                "historical_status": report["overall_status"],
                "fixture_provenance": "adapter_synthetic"
                if original_manifest["scenarios"].get("fixture")
                else "unverified_adapter_fixture",
                "manifest": original_manifest,
            }
        )
    return found, exclusions
