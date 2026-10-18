"""Catch the observed timm versus torchvision RandomErasing API mix-up."""

import importlib.machinery
import importlib.util

from Preflight_checker.preflight import check_source


def test_timm_random_erasing_rejects_torchvision_p_keyword(monkeypatch):
    """The real A1 node failed before training because timm expects probability."""
    original_find_spec = importlib.util.find_spec

    def available_timm(module):
        if module == "timm":
            return importlib.machinery.ModuleSpec("timm", loader=None)
        return original_find_spec(module)

    monkeypatch.setattr(importlib.util, "find_spec", available_timm)
    source = (
        "from timm.data.random_erasing import RandomErasing\n"
        "train_tf = RandomErasing(p=0.25, mode='pixel')\n"
    )

    report = check_source(source, shared_memory_bytes=1 << 40)

    assert report["admitted"] is False
    assert "timm_random_erasing_invalid_keyword" in report["issues"]
    assert any(
        item["code"] == "timm_random_erasing_invalid_keyword"
        and item["line"] == 2
        and "probability" in item["message"]
        for item in report["diagnostics"]
    )
