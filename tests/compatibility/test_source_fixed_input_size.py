"""Regression for the Petfinder candidate's Swin 224 versus 384-pixel input."""

import importlib.machinery
import importlib.util

import pytest

from Preflight_checker.preflight import check_source


SOURCE = """
import timm
from torchvision import transforms
from torch.utils.data import DataLoader

IMG = 384
tfm_train = transforms.Compose([
    transforms.RandomResizedCrop(IMG),
    transforms.ToTensor(),
])

class PetDS:
    def __init__(self, transform):
        self.transform = transform

train_loader = DataLoader(PetDS(tfm_train), batch_size=32)

class PetModel:
    def __init__(self, arch):
        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0)

known = ["swin_tiny_patch4_window7_224", "convnext_tiny"]
jobs = [(arch, seed) for arch in known for seed in (42, 777)]
for arch, seed in jobs:
    model = PetModel(arch)
"""


@pytest.fixture(autouse=True)
def _timm_available(monkeypatch):
    original_find_spec = importlib.util.find_spec

    def find_spec(module):
        if module == "timm":
            return importlib.machinery.ModuleSpec("timm", loader=None)
        return original_find_spec(module)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)


def test_rejects_loop_selected_fixed_224_model_with_384_pixel_transform():
    report = check_source(SOURCE, shared_memory_bytes=0)

    assert report["admitted"] is False
    assert "fixed_input_image_size_mismatch" in report["issues"]
    assert any(
        item["code"] == "fixed_input_image_size_mismatch"
        and "swin_tiny_patch4_window7_224" in item["message"]
        and "384" in item["message"]
        for item in report["diagnostics"]
    )


def test_rejects_suffix_named_swin_selected_by_fold_index():
    source = '''
import timm
from torchvision import transforms as T
from torch.utils.data import DataLoader
IMG_SIZE = 384
train_tf = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
class PetDataset:
    def __init__(self, transform):
        self.transform = transform
    def __getitem__(self, index):
        image = Image.open("example.jpg").convert("RGB")
        return self.transform(image)
train_ds = PetDataset(train_tf)
class PetModel:
    def __init__(self, backbone_candidates):
        self.backbone = _create_backbone(backbone_candidates)
def _create_backbone(candidates):
    for name in candidates:
        return timm.create_model(name, pretrained=True, num_classes=0)
BACKBONES = ["convnext_tiny.fb_in22k_ft_in1k", "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"]
for fold in range(2):
    backbone_name = BACKBONES[fold % len(BACKBONES)]
    train_loader = DataLoader(train_ds, batch_size=32)
    model = PetModel([backbone_name])
'''
    report = check_source(source, shared_memory_bytes=0)

    assert report["admitted"] is False
    assert "fixed_input_image_size_mismatch" in report["issues"]
    assert any(
        "swin_tiny_patch4_window7_224.ms_in22k_ft_in1k" in item["message"]
        for item in report["diagnostics"]
    )


def test_does_not_reject_unused_suffix_named_swin():
    source = SOURCE.replace(
        'known = ["swin_tiny_patch4_window7_224", "convnext_tiny"]',
        'unused = ["swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"]\n'
        'known = ["convnext_tiny"]',
    )

    assert check_source(source, shared_memory_bytes=0)["admitted"] is True


def test_accepts_matching_224_pixel_transform():
    report = check_source(SOURCE.replace("IMG = 384", "IMG = 224"), shared_memory_bytes=0)

    assert report["admitted"] is True


def test_accepts_384_pixel_transform_without_fixed_224_model():
    report = check_source(
        SOURCE.replace('"swin_tiny_patch4_window7_224", ', ""),
        shared_memory_bytes=0,
    )

    assert report["admitted"] is True


def test_rejects_observed_helper_and_concatenated_model_jobs():
    source = SOURCE.replace(
        "jobs = [(arch, seed) for arch in known for seed in (42, 777)]\n"
        "for arch, seed in jobs:\n"
        "    model = PetModel(arch)",
        "extra = [\"swin_small_patch4_window7_224\", \"resnet101\"]\n"
        "jobs = [(arch, seed) for arch in known for seed in (42, 777)] + "
        "[(arch, 42) for arch in extra]\n"
        "def run(arch, seed):\n"
        "    model = PetModel(arch)\n"
        "for arch, seed in jobs:\n"
        "    run(arch, seed)",
    )

    report = check_source(source, shared_memory_bytes=0)

    assert report["admitted"] is False
    assert "fixed_input_image_size_mismatch" in report["issues"]


def test_ignores_unused_fixed_size_model_name():
    source = SOURCE.replace(
        'known = ["swin_tiny_patch4_window7_224", "convnext_tiny"]',
        'unused = ["swin_tiny_patch4_window7_224"]\nknown = ["convnext_tiny"]',
    )

    report = check_source(source, shared_memory_bytes=0)

    assert report["admitted"] is True


def test_ignores_unused_mismatched_transform():
    source = SOURCE.replace(
        "IMG = 384\ntfm_train = transforms.Compose([",
        "IMG = 384\nunused_tfm = transforms.Compose([transforms.RandomResizedCrop(IMG)])\n"
        "tfm_train = transforms.Compose([",
    ).replace(
        "transforms.RandomResizedCrop(IMG),\n    transforms.ToTensor(),",
        "transforms.RandomResizedCrop(224),\n    transforms.ToTensor(),",
    )

    report = check_source(source, shared_memory_bytes=0)

    assert report["admitted"] is True


def test_rejects_first_dinov2_fallback_with_336_pixel_loader_input():
    source = '''
import timm
from torchvision import transforms as T
from torch.utils.data import DataLoader

IMG_SIZE = 336
BACKBONE_CANDIDATES = [
    "vit_base_patch14_dinov2.lvd142m",
    "convnext_base.fb_in22k_ft_in1k",
]
train_tf = T.Compose([T.RandomResizedCrop(IMG_SIZE), T.ToTensor()])

class PetDataset:
    def __init__(self, transform):
        self.transform = transform
    def __getitem__(self, index):
        image = Image.open("example.jpg").convert("RGB")
        return self.transform(image)

def create_backbone():
    for model_name in BACKBONE_CANDIDATES:
        try:
            model = timm.create_model(model_name, pretrained=True, num_classes=0)
            return model, model_name
        except Exception:
            model = timm.create_model(model_name, pretrained=False, num_classes=0)
            return model, model_name

class PetModel:
    def __init__(self):
        self.backbone, self.backbone_name = create_backbone()

train_loader = DataLoader(PetDataset(train_tf), batch_size=32)
model = PetModel()
'''

    report = check_source(source, shared_memory_bytes=1 << 30)

    assert "fixed_input_image_size_mismatch" in report["issues"]
    assert any(
        item["code"] == "fixed_input_image_size_mismatch"
        and "vit_base_patch14_dinov2.lvd142m" in item["message"]
        and "518" in item["message"]
        and "336" in item["message"]
        for item in report["diagnostics"]
    )


def test_rejects_dinov2_mismatch_with_cv2_image_dataset():
    source = '''
import timm
import cv2
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import DataLoader

IMG_SIZE = 336
BACKBONE_CANDIDATES = ["vit_base_patch14_dinov2.lvd142m", "convnext_tiny"]

class PetDataset:
    def __init__(self, transform=None):
        self.transform = transform
    def __getitem__(self, index):
        image = cv2.imread("example.jpg", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        return self.transform(image)

def create_backbone():
    for model_name in BACKBONE_CANDIDATES:
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        return model, model_name

def run_training():
    train_tf = T.Compose([T.RandomResizedCrop(IMG_SIZE), T.ToTensor()])
    train_ds = PetDataset(transform=train_tf)
    train_loader = DataLoader(train_ds, batch_size=32)
    model, model_name = create_backbone()
'''

    report = check_source(source, shared_memory_bytes=1 << 30)

    assert "fixed_input_image_size_mismatch" in report["issues"]


def test_ignores_dinov2_factory_that_is_never_called():
    source = '''
import timm
from torchvision import transforms as T
from torch.utils.data import DataLoader
IMG_SIZE = 336
BACKBONE_CANDIDATES = ["vit_base_patch14_dinov2.lvd142m"]

class PetDataset:
    def __init__(self, transform):
        self.transform = transform
    def __getitem__(self, index):
        image = Image.open("example.jpg").convert("RGB")
        return self.transform(image)

def unused_backbone_factory():
    for model_name in BACKBONE_CANDIDATES:
        model = timm.create_model(model_name, pretrained=False)
        return model

train_tf = T.Compose([T.RandomResizedCrop(IMG_SIZE), T.ToTensor()])
train_loader = DataLoader(PetDataset(train_tf), batch_size=32)
model = timm.create_model("convnext_tiny", pretrained=False)
'''

    report = check_source(source, shared_memory_bytes=1 << 30)

    assert "fixed_input_image_size_mismatch" not in report["issues"]


def test_accepts_fallback_factory_that_probes_actual_input_before_return():
    """Catch the false rejection of A1 node 20's guarded DINOv2 fallback."""
    source = '''
import timm
from torchvision import transforms as T
from torch.utils.data import DataLoader

IMG_SIZE = 336
BACKBONE_CANDIDATES = [
    "vit_base_patch14_dinov2.lvd142m",
    "convnext_base.fb_in22k_ft_in1k",
]
train_tf = T.Compose([T.RandomResizedCrop(IMG_SIZE), T.ToTensor()])

class PetDataset:
    def __init__(self, transform):
        self.transform = transform
    def __getitem__(self, index):
        return self.transform(Image.open("example.jpg").convert("RGB"))

def _probe_backbone(model, size=IMG_SIZE):
    probe = torch.zeros(1, 3, size, size)
    model.forward_features(probe)

def create_backbone():
    for model_name in BACKBONE_CANDIDATES:
        for extra in ({"dynamic_img_size": True}, {}):
            try:
                model = timm.create_model(model_name, pretrained=True, **extra)
                _probe_backbone(model, IMG_SIZE)
                return model, model_name
            except Exception:
                continue

train_loader = DataLoader(PetDataset(train_tf), batch_size=32)
model, model_name = create_backbone()
'''

    report = check_source(source, shared_memory_bytes=1 << 30)

    assert "fixed_input_image_size_mismatch" not in report["issues"]
    unsafe_source = source.replace("model.forward_features(probe)", "pass")
    unsafe_report = check_source(unsafe_source, shared_memory_bytes=1 << 30)
    assert "fixed_input_image_size_mismatch" in unsafe_report["issues"]
    swallowed_probe = source.replace(
        "    model.forward_features(probe)",
        "    try:\n        model.forward_features(probe)\n"
        "    except Exception:\n        pass",
    )
    swallowed_report = check_source(swallowed_probe, shared_memory_bytes=1 << 30)
    assert "fixed_input_image_size_mismatch" in swallowed_report["issues"]
