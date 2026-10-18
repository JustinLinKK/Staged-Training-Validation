"""Regression for the observed one-hour A10G multi-fold training timeout."""

import subprocess

from Preflight_checker import preflight


SOURCE = """
BACKBONE_NAME = 'convnext_small.fb_in22k_ft_in1k'
IMG_SIZE = 384
N_FOLDS = 5
BATCH_SIZE = 16
NUM_EPOCHS = 12

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.ToTensor(),
])

class PetDataset:
    def __getitem__(self, index):
        return train_tf(Image.open(path).convert('RGB'))

class PetModel:
    def __init__(self):
        self.backbone = timm.create_model(BACKBONE_NAME, pretrained=True)

for fold in range(N_FOLDS):
    loader = DataLoader(PetDataset(), batch_size=BATCH_SIZE, shuffle=True)
    model = PetModel()
    for epoch in range(NUM_EPOCHS):
        for image in loader:
            loss = model.backbone(image)
            loss.backward()
"""


def test_rejects_observed_a10g_five_fold_convnext_runtime_risk(monkeypatch):
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="NVIDIA A10G, 23028\n", stderr=""
        ),
    )

    report = preflight.check_source(SOURCE, shared_memory_bytes=0)

    assert report["admitted"] is False
    assert "gpu_runtime_budget_known_configuration" in report["issues"]


def test_rejects_backbone_passed_through_constructor_default(monkeypatch):
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="NVIDIA A10G, 23028\n", stderr=""
        ),
    )
    source = SOURCE.replace(
        "def __init__(self):\n        self.backbone = timm.create_model(BACKBONE_NAME, pretrained=True)",
        "def __init__(self, backbone_name=BACKBONE_NAME):\n"
        "        self.backbone = timm.create_model(backbone_name, pretrained=True)",
    )

    report = preflight.check_source(source, shared_memory_bytes=0)

    assert "gpu_runtime_budget_known_configuration" in report["issues"]


def test_longer_execution_budget_does_not_claim_observed_one_hour_failure(monkeypatch):
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="NVIDIA A10G, 23028\n", stderr=""
        ),
    )

    report = preflight.check_source(
        SOURCE, shared_memory_bytes=0, execution_timeout_seconds=7200
    )

    assert "gpu_runtime_budget_known_configuration" not in report["issues"]


def test_a100_is_not_rejected_by_a10g_observation(monkeypatch):
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="NVIDIA A100-SXM4-80GB, 81920\n", stderr=""
        ),
    )

    report = preflight.check_source(SOURCE, shared_memory_bytes=0)

    assert "gpu_runtime_budget_known_configuration" not in report["issues"]
