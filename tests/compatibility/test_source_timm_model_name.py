"""Reject literal timm backbone names absent from the candidate environment."""

import importlib.machinery
import importlib.util
import sys
from types import SimpleNamespace

from Preflight_checker.preflight import check_source


def test_enumerated_backbone_list_rejects_unknown_timm_model(monkeypatch):
    original_find_spec = importlib.util.find_spec

    def available_timm(module):
        if module == "timm":
            return importlib.machinery.ModuleSpec("timm", loader=None)
        return original_find_spec(module)

    monkeypatch.setattr(importlib.util, "find_spec", available_timm)
    monkeypatch.setitem(
        sys.modules,
        "timm",
        SimpleNamespace(is_model=lambda model: model == "convnext_tiny"),
    )
    source = """\
import timm
BACKBONES = ["convnext_tiny", "swin_tiny"]
class CandidateModel:
    def __init__(self, backbone):
        self.model = timm.create_model(backbone, pretrained=False)
def build_model(backbone_name):
    return CandidateModel(backbone=backbone_name)
for index, backbone in enumerate(BACKBONES):
    model = build_model(backbone)
"""

    report = check_source(source, shared_memory_bytes=1 << 40)

    assert report["admitted"] is False
    assert "timm_unknown_model" in report["issues"]
    assert any(
        item["code"] == "timm_unknown_model"
        and item["line"] == 2
        and "swin_tiny" in item["message"]
        for item in report["diagnostics"]
    )


def test_unused_backbone_list_does_not_trigger_model_name_check(monkeypatch):
    original_find_spec = importlib.util.find_spec

    def available_timm(module):
        if module == "timm":
            return importlib.machinery.ModuleSpec("timm", loader=None)
        return original_find_spec(module)

    monkeypatch.setattr(importlib.util, "find_spec", available_timm)
    monkeypatch.setitem(
        sys.modules, "timm", SimpleNamespace(is_model=lambda model: model == "valid_literal")
    )
    source = """\
import timm
BACKBONES = ["swin_tiny"]
model = timm.create_model("valid_literal", pretrained=False)
"""

    report = check_source(source, shared_memory_bytes=1 << 40)

    assert report["admitted"] is True


def test_unrelated_loop_does_not_feed_timm_model(monkeypatch):
    original_find_spec = importlib.util.find_spec

    def available_timm(module):
        if module == "timm":
            return importlib.machinery.ModuleSpec("timm", loader=None)
        return original_find_spec(module)

    monkeypatch.setattr(importlib.util, "find_spec", available_timm)
    monkeypatch.setitem(
        sys.modules,
        "timm",
        SimpleNamespace(is_model=lambda model: model == "valid_literal"),
    )
    source = """\
import timm
BACKBONES = ["swin_tiny"]
model = timm.create_model("valid_literal", pretrained=False)
for backbone in BACKBONES:
    print(backbone)
"""

    report = check_source(source, shared_memory_bytes=1 << 40)

    assert report["admitted"] is True
