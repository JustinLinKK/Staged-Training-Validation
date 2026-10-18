"""Regression for an executed Petfinder A10G candidate that timed out."""

import subprocess

from Preflight_checker import preflight


SOURCE = """
BACKBONES = [
    'convnext_small.fb_in22k_ft_in1k',
    'swin_small_patch4_window7_224',
]
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 7
PATIENCE = 3
target_col = 'Pawpularity'
split_info = {'n_folds': 5}
si = split_info
n_folds = int(si.get('n_folds', 5))
train_tf = transforms.RandomResizedCrop(IMG_SIZE)

def build_loaders():
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def build_model(backbone_name):
    return timm.create_model(backbone_name, pretrained=True)

def train_one_epoch(model, train_loader):
    for image in train_loader:
        loss = model(image).sum()
        loss.backward()

for fold in range(n_folds):
    train_loader = build_loaders()
    for bb_i, bb in enumerate(BACKBONES):
        model = build_model(bb)
        no_improve = 0
        for epoch in range(min(int(_MAX_EPOCHS), EPOCHS) if _PROBE else EPOCHS):
            train_one_epoch(model, train_loader)
            if no_improve >= PATIENCE:
                break
"""


def test_rejects_observed_petfinder_two_backbone_hour_budget(monkeypatch):
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="NVIDIA A10G, 23028\n", stderr=""
        ),
    )

    report = preflight.check_source(
        SOURCE, shared_memory_bytes=1 << 40, execution_timeout_seconds=3600
    )

    assert "gpu_runtime_budget_known_configuration" in report["issues"]


def test_does_not_reject_two_fold_override(monkeypatch):
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="NVIDIA A10G, 23028\n", stderr=""
        ),
    )
    source = SOURCE.replace("n_folds = int(si.get('n_folds', 5))", "n_folds = 2")

    report = preflight.check_source(
        source, shared_memory_bytes=1 << 40, execution_timeout_seconds=3600
    )

    assert "gpu_runtime_budget_known_configuration" not in report["issues"]
