import os
from pathlib import Path

import pytest
import yaml

from model_preflight.api import check


def _manifest(
    tmp_path: Path,
    source: str,
    timeout: float = 4.0,
    *,
    allow_real_cpu_abstract_fallback: bool = False,
) -> Path:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "adapter.py").write_text(source, encoding="utf-8")
    manifest = tmp_path / "preflight.yaml"
    manifest.write_text(
        f"""
schema_version: 1
candidate: {{id: isolation, root: ./candidate, adapter: adapter:CandidateAdapter}}
task: {{name: test}}
scenarios: {{train_batch_sizes: [1], test_last_batch: false, run_validation: false}}
target: {{profile: nvidia/a10_24gb}}
execution:
  cpu_timeout_seconds: {timeout}
  allow_real_cpu_abstract_fallback: {str(allow_real_cpu_abstract_fallback).lower()}
  maximum_cpu_memory_mb: 4096
""",
        encoding="utf-8",
    )
    return manifest


def test_timeout_is_inconclusive(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        """
import time
time.sleep(30)
class CandidateAdapter:
    pass
""",
        timeout=1.0,
    )
    report = check(manifest, only={"construction"})
    assert report.overall_status.value == "INCONCLUSIVE"
    assert report.diagnostics[0].code == "CHK002"


def test_worker_crash_is_internal_error(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        """
import os
os._exit(17)
""",
    )
    report = check(manifest, only={"construction"})
    assert report.overall_status.value == "INTERNAL_ERROR"
    assert report.diagnostics[0].code == "CHK003"


def test_configured_real_cpu_abstract_fallback(tmp_path: Path) -> None:
    manifest = _manifest(
        tmp_path,
        """
import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 1)
    def forward(self, value):
        if value.is_meta or type(value).__name__ == "FakeTensor":
            raise NotImplementedError("no abstract kernel")
        return self.layer(value)

class CandidateAdapter:
    def build_model(self, context):
        return Model()
    def build_optimizer(self, model, context):
        return torch.optim.SGD(model.parameters(), lr=0.1)
    def build_train_batch(self, scenario, device):
        return torch.randn(scenario["batch_size"], 4, device=device), None
    def training_step(self, model, batch, context):
        return model(batch[0]).mean()
""",
        allow_real_cpu_abstract_fallback=True,
    )
    report = check(manifest, only={"abstract_forward"})
    assert report.overall_status.value == "PASS", report.to_dict()
    scenario = report.stages[0].evidence["scenarios"][0]
    assert scenario["backend"] == "real_cpu_fallback"


def test_parallel_workers_apply_limits_and_threads_before_candidate_import(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, """
import os
import resource
import torch
from torch import nn

assert torch.get_num_threads() == 2
assert torch.get_num_interop_threads() == 1
assert os.environ['OMP_NUM_THREADS'] == '2'
assert os.environ['OPENBLAS_NUM_THREADS'] == '2'
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
assert resource.getrlimit(resource.RLIMIT_AS) == (4096 * 1024**2,) * 2
assert resource.getrlimit(resource.RLIMIT_CPU) == (30, 31)
assert resource.getrlimit(resource.RLIMIT_CORE) == (0, 0)
assert resource.getrlimit(resource.RLIMIT_FSIZE) == (1000000, 1000000)
if os.getuid() != 0:
    assert resource.getrlimit(resource.RLIMIT_NPROC) == (4096, 4096)
print(os.getpid(), flush=True)

class CandidateAdapter:
    def build_model(self, context):
        return nn.Linear(4, 1)
    def build_optimizer(self, model, context):
        return torch.optim.SGD(model.parameters(), lr=0.1)
    def build_train_batch(self, scenario, device):
        return torch.randn(scenario['batch_size'], 4, device=device), None
    def training_step(self, model, batch, context):
        return model(batch[0]).mean()
""")
    raw = yaml.safe_load(manifest.read_text())
    raw["execution"].update(
        max_workers=2, cpu_threads=2, abstract_timeout_seconds=15, maximum_processes=4096,
    )
    raw["scenarios"]["train_batch_sizes"] = [1, 2, 3, 4]
    manifest.write_text(yaml.safe_dump(raw))
    report = check(manifest, only={"abstract_forward"})
    assert report.overall_status.value == "PASS", report.to_dict()
    scenarios = report.stages[0].evidence["scenarios"]
    assert len(scenarios) == 4
    assert len({s["captured_stdout"].strip() for s in scenarios}) == 4
    assert all(s["cpu_threads"] == 2 for s in scenarios)


def test_parallel_timeout_does_not_discard_healthy_scenario(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, """
import time
import torch
from torch import nn

class CandidateAdapter:
    def build_model(self, context):
        if context['scenario']['batch_size'] == 1:
            time.sleep(30)
        return nn.Linear(4, 1)
    def build_optimizer(self, model, context):
        return torch.optim.SGD(model.parameters(), lr=0.1)
    def build_train_batch(self, scenario, device):
        return torch.randn(scenario['batch_size'], 4, device=device), None
    def training_step(self, model, batch, context):
        return model(batch[0]).mean()
""")
    raw = yaml.safe_load(manifest.read_text())
    raw["execution"].update(max_workers=2, abstract_timeout_seconds=6)
    raw["scenarios"]["train_batch_sizes"] = [1, 2]
    manifest.write_text(yaml.safe_dump(raw))
    report = check(manifest, only={"abstract_forward"})
    assert report.overall_status.value == "INCONCLUSIVE", report.to_dict()
    scenarios = report.stages[0].evidence["scenarios"]
    assert [s["status"] for s in scenarios] == ["INCONCLUSIVE", "PASS"]
    assert scenarios[0]["termination_cause"] == "wall_timeout"
    assert [d.code for d in report.diagnostics] == ["CHK002"]


@pytest.mark.skipif(os.getuid() == 0, reason="RLIMIT_NPROC does not restrict root")
def test_exhausted_thread_budget_is_inconclusive_before_import(tmp_path: Path) -> None:
    marker = tmp_path / "imported"
    manifest = _manifest(
        tmp_path, f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
    )
    raw = yaml.safe_load(manifest.read_text())
    raw["execution"].update(cpu_threads=2, maximum_processes=1)
    manifest.write_text(yaml.safe_dump(raw))
    report = check(manifest, only={"construction"})
    assert report.overall_status.value == "INCONCLUSIVE", report.to_dict()
    assert [d.code for d in report.diagnostics] == ["CHK001"]
    assert "maximum_processes" in report.diagnostics[0].message
    assert not marker.exists()
