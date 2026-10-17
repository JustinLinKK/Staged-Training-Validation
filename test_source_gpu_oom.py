"""Regression for the observed A10G ConvNeXt-Base first-batch OOM."""

import importlib.machinery
import importlib.util
import subprocess

import pytest

from Preflight_checker.preflight import check_source


SOURCE = """
import timm
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

IMG_SIZE = 384
BATCH_SIZE = 64
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.ToTensor(),
])

class PetDataset:
    def __init__(self, transform):
        self.transform = transform
    def __getitem__(self, index):
        return self.transform(Image.open(path).convert("RGB"))

train_ds = PetDataset(train_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

class PetModel:
    def __init__(self, backbone_name):
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
    def __call__(self, images):
        return self.backbone(images)

backbone_candidates = ["convnext_base_384_in22ft1k", "convnext_base"]
for backbone_name in backbone_candidates:
    model = PetModel(backbone_name)
    break
for images in train_loader:
    loss = model(images)
    loss.backward()
"""


def _a10g_query(*args, **kwargs):
    return subprocess.CompletedProcess(args, 0, stdout="NVIDIA A10G, 23028\n", stderr="")


@pytest.fixture(autouse=True)
def _timm_available(monkeypatch):
    """Keep GPU-fingerprint cases independent of optional local packages."""
    original_find_spec = importlib.util.find_spec

    def find_spec(module):
        if module == "timm":
            return importlib.machinery.ModuleSpec("timm", loader=None)
        return original_find_spec(module)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def test_rejects_observed_convnext_384_batch_64_oom_on_a10g(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _a10g_query)

    report = check_source(SOURCE, shared_memory_bytes=2 * 1024**3)

    assert report["admitted"] is False
    assert "gpu_oom_known_configuration" in report["issues"]
    assert report["assurance"] == "bounded_resource_risk"


def test_does_not_reject_smaller_image_from_oom_fingerprint(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _a10g_query)

    report = check_source(SOURCE.replace("IMG_SIZE = 384", "IMG_SIZE = 256"), shared_memory_bytes=2 * 1024**3)

    assert report["admitted"] is True


def test_does_not_reject_large_memory_gpu_from_oom_fingerprint(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="NVIDIA A100 80GB PCIe, 81920\n", stderr=""
        ),
    )

    report = check_source(SOURCE, shared_memory_bytes=2 * 1024**3)

    assert report["admitted"] is True
