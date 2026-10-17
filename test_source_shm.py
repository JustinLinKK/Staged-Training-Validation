"""Bounded checks for the Petfinder DataLoader shared-memory failure."""

import subprocess
import sys
from textwrap import indent
from pathlib import Path

from Preflight_checker.preflight import check_source


PETFINDER_WORKER_PATTERN = """
from torch.utils.data import DataLoader
from torchvision import transforms

batch_size = 32
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
])
image = Image.open(path).convert("RGB")
train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8)
"""


def test_rejects_image_loader_whose_prefetch_exceeds_container_shm() -> None:
    report = check_source(
        PETFINDER_WORKER_PATTERN,
        shared_memory_bytes=64 * 1024 * 1024,
    )

    assert report["admitted"] is False
    assert "dataloader_shm_risk" in report["issues"]
    assert report["assurance"] == "bounded_resource_risk"
    assert "64 MiB" in report["diagnostics"][0]["message"]


def test_same_candidate_admitted_with_adequate_shared_memory() -> None:
    report = check_source(
        PETFINDER_WORKER_PATTERN,
        shared_memory_bytes=2 * 1024 * 1024 * 1024,
    )

    assert report["admitted"] is True


def test_tiny_loader_does_not_get_rejected_by_resource_heuristic() -> None:
    source = PETFINDER_WORKER_PATTERN.replace("batch_size = 32", "batch_size = 1").replace(
        "num_workers=8", "num_workers=1"
    )
    report = check_source(source, shared_memory_bytes=64 * 1024 * 1024)

    assert report["admitted"] is True


def test_rejects_exhausted_shared_memory() -> None:
    report = check_source(PETFINDER_WORKER_PATTERN, shared_memory_bytes=0)

    assert report["admitted"] is False


def test_rejects_known_batch_under_environment_override() -> None:
    source = PETFINDER_WORKER_PATTERN.replace(
        "batch_size=batch_size, num_workers=8",
        "batch_size=int(override) if override is not None else batch_size, num_workers=8",
    )
    report = check_source(source, shared_memory_bytes=64 * 1024 * 1024)

    assert report["admitted"] is False


def test_rejects_local_batch_assignment_variant_of_buggy_node() -> None:
    source = "def main():\n" + indent(PETFINDER_WORKER_PATTERN, "    ")

    report = check_source(source, shared_memory_bytes=64 * 1024 * 1024)

    assert "dataloader_shm_risk" in report["issues"]


def test_rejects_fixed_size_resize_without_crop() -> None:
    source = PETFINDER_WORKER_PATTERN.replace(
        "transforms.RandomResizedCrop(224)", "transforms.Resize((224, 224))"
    )

    report = check_source(source, shared_memory_bytes=64 * 1024 * 1024)

    assert "dataloader_shm_risk" in report["issues"]


def test_uses_final_crop_size_instead_of_earlier_resize() -> None:
    source = PETFINDER_WORKER_PATTERN.replace("batch_size = 32", "batch_size = 16")
    source = source.replace(
        "transforms.RandomResizedCrop(224)",
        "transforms.Resize((256, 256)),\n    transforms.CenterCrop(224)",
    )

    report = check_source(source, shared_memory_bytes=80 * 1024 * 1024)

    assert report["admitted"] is True


def test_rejects_cfg_class_image_size_in_generated_node() -> None:
    source = """
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as T
class CFG:
    img_size = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
transform = T.Compose([T.RandomResizedCrop(CFG.img_size), T.ToTensor()])
image = Image.open(path).convert("RGB")
loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
"""

    report = check_source(source, shared_memory_bytes=64 * 1024 * 1024)

    assert "dataloader_shm_risk" in report["issues"]


def test_uses_batch_value_at_loader_instead_of_later_reassignment() -> None:
    source = PETFINDER_WORKER_PATTERN + "\nbatch_size = 1\n"

    report = check_source(source, shared_memory_bytes=64 * 1024 * 1024)

    assert "dataloader_shm_risk" in report["issues"]


def test_does_not_pair_unrelated_image_pipeline_with_tabular_loader() -> None:
    source = PETFINDER_WORKER_PATTERN.replace(
        "train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8)",
        "class TabularDataset:\n"
        "    def __getitem__(self, index):\n"
        "        return torch.tensor([index])\n"
        "tabular_dataset = TabularDataset()\n"
        "train_loader = DataLoader(tabular_dataset, batch_size=batch_size, num_workers=8)",
    )

    report = check_source(source, shared_memory_bytes=64 * 1024 * 1024)

    assert report["admitted"] is True


def test_does_not_reject_numpy_backed_tabular_loader_for_unrelated_image_helper() -> None:
    source = """
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
image_transform = transforms.Compose([transforms.RandomResizedCrop(512), transforms.ToTensor()])
def unused_image_helper(path):
    return image_transform(Image.open(path).convert("RGB"))
class TabularDataset:
    def __getitem__(self, index):
        return torch.tensor(np.asarray([index], dtype=np.float32))
dataset = TabularDataset()
loader = DataLoader(dataset, batch_size=32, num_workers=8)
"""

    report = check_source(source, shared_memory_bytes=64 * 1024 * 1024)

    assert report["admitted"] is True


def test_does_not_reject_tensor_dataset_for_unrelated_image_helper() -> None:
    source = """
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from PIL import Image
unused_transform = transforms.Compose([transforms.RandomResizedCrop(512), transforms.ToTensor()])
def unused_image_helper(path):
    return unused_transform(Image.open(path).convert("RGB"))
dataset = TensorDataset(torch.zeros(100, 4), torch.zeros(100))
loader = DataLoader(dataset, batch_size=32, num_workers=8)
"""

    report = check_source(source, shared_memory_bytes=64 * 1024 * 1024)

    assert report["admitted"] is True


def test_uses_the_loaded_dataset_transform_not_a_different_image_pipeline() -> None:
    source = """
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
class ImageDataset:
    def __init__(self, transform):
        self.transform = transform
    def __getitem__(self, index):
        image = Image.open(path).convert("RGB")
        return self.transform(image)
small = transforms.Compose([transforms.RandomResizedCrop(64), transforms.ToTensor()])
large = transforms.Compose([transforms.RandomResizedCrop(512), transforms.ToTensor()])
train_dataset = ImageDataset(transform=small)
loader = DataLoader(train_dataset, batch_size=32, num_workers=8)
"""

    report = check_source(source, shared_memory_bytes=64 * 1024 * 1024)

    assert report["admitted"] is True


def test_uses_positional_dataset_transform_not_an_unrelated_large_pipeline() -> None:
    source = """
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
class ImageDataset:
    def __init__(self, frame, transform):
        self.transform = transform
    def __getitem__(self, index):
        return self.transform(Image.open(path).convert("RGB"))
small = transforms.Compose([transforms.RandomResizedCrop(64), transforms.ToTensor()])
large = transforms.Compose([transforms.RandomResizedCrop(512), transforms.ToTensor()])
dataset = ImageDataset(frame, small)
loader = DataLoader(dataset, batch_size=32, num_workers=8)
"""

    report = check_source(source, shared_memory_bytes=64 * 1024 * 1024)

    assert report["admitted"] is True


def test_missing_default_manifest_reports_actionable_cli_error() -> None:
    script = Path(__file__).with_name("preflight.py")

    result = subprocess.run(
        [sys.executable, str(script), "--json"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "supply node paths" in result.stderr
    assert "Traceback" not in result.stderr


def test_preloaded_image_tensors_are_not_misclassified_as_tabular() -> None:
    source = PETFINDER_WORKER_PATTERN.replace(
        "train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8)",
        "class ImageDataset:\n"
        "    def __init__(self, transform):\n"
        "        self.images = [transform(Image.open(path).convert('RGB'))]\n"
        "    def __getitem__(self, index):\n"
        "        return self.images[index]\n"
        "dataset = ImageDataset(transform=train_transform)\n"
        "train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=8)",
    )

    report = check_source(source, shared_memory_bytes=64 * 1024 * 1024)

    assert "dataloader_shm_risk" in report["issues"]


def test_images_preloaded_outside_dataset_are_not_misclassified_as_tabular() -> None:
    source = """
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.ToTensor()])
images = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths])
class ImageDataset:
    def __init__(self, images):
        self.images = images
    def __getitem__(self, index):
        return self.images[index]
dataset = ImageDataset(images)
loader = DataLoader(dataset, batch_size=32, num_workers=8)
"""

    report = check_source(source, shared_memory_bytes=64 * 1024 * 1024)

    assert "dataloader_shm_risk" in report["issues"]
