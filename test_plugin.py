"""The portable preflight protocol must not silently admit untested candidates."""

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from Preflight_checker.plugin import ProbeSpec, SpeculativePreflight, _cpu_probe, check_request, probe_from_environment, submit_with_speculation


def test_static_rejection_never_submits(tmp_path: Path) -> None:
    submitted = []
    result = SpeculativePreflight().run(
        "def broken(:\n", workspace=tmp_path, node_id="bad",
        submit=lambda: submitted.append("job") or "job",
        cancel=lambda job_id: None,
        probe=ProbeSpec((sys.executable, "-c", "print('never')"), tmp_path),
    )
    assert result["status"] == "static_rejected"
    assert submitted == []


def test_two_epoch_probe_runs_after_speculative_submit(tmp_path: Path) -> None:
    marker = tmp_path / "submitted"
    script = (
        "import json,os,pathlib; "
        f"assert pathlib.Path({str(marker)!r}).exists(); "
        "assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'; "
        "assert os.environ['LIGHTNING_CPU_EPOCHS'] == '2'; "
        "print(json.dumps({'epochs_completed': 2, 'status': 'passed'}))"
    )
    cancelled = []

    def submit() -> str:
        marker.write_text("yes")
        return "job-1"

    result = SpeculativePreflight().run(
        "print('candidate')", workspace=tmp_path, node_id="good",
        submit=submit, cancel=cancelled.append,
        probe=ProbeSpec((sys.executable, "-c", script), tmp_path),
    )
    assert result["status"] == "probe_passed"
    assert result["job_id"] == "job-1"
    assert result["probe"]["epochs_completed"] == 2
    assert cancelled == []


def test_only_confirmed_cpu_bug_cancels_provisional_job(tmp_path: Path) -> None:
    for output, expected_status, expected_cancel in (
        ({"status": "failed", "bug_confirmed": True, "epochs_completed": 1}, "probe_rejected", ["job-2"]),
        ({"status": "failed", "epochs_completed": 1}, "probe_inconclusive", []),
        ({"status": "passed", "epochs_completed": 1}, "probe_inconclusive", []),
    ):
        cancelled = []
        result = SpeculativePreflight().run(
            "print('candidate')", workspace=tmp_path, node_id="buggy",
            submit=lambda: "job-2", cancel=cancelled.append,
            probe=ProbeSpec(
                (sys.executable, "-c", f"print({json.dumps(json.dumps(output))})"),
                tmp_path,
            ),
        )
        assert result["status"] == expected_status
        assert cancelled == expected_cancel
        if expected_status == "probe_rejected":
            report = json.loads((tmp_path / "preflight" / "buggy" / "report.json").read_text())
            assert report["admitted"] is False
            assert report["issues"] == ["cpu_probe_confirmed_bug"]


def test_protocol_checks_source_without_executing_it(tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    report = check_request({
        "version": 1,
        "source": f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')",
        "filename": "node.py",
        "working_directory": str(tmp_path),
    })
    assert report["admitted"] is True
    assert not marker.exists()


def test_protocol_uses_caller_shared_memory_not_small_container_default(tmp_path: Path) -> None:
    source = (
        "from torch.utils.data import DataLoader\n"
        "from torchvision import transforms\n"
        "batch_size = 32\n"
        "train_transform = transforms.Compose([\n"
        "    transforms.RandomResizedCrop(224),\n"
        "    transforms.ToTensor(),\n"
        "])\n"
        "image = Image.open(path).convert('RGB')\n"
        "loader = DataLoader(dataset, batch_size=batch_size, num_workers=8)\n"
    )
    request = {"version": 1, "source": source, "working_directory": str(tmp_path),
               "available_modules": ["torch", "torchvision"],
               "shared_memory_bytes": 2 * 1024**3}
    assert check_request(request)["admitted"] is True
    request["shared_memory_bytes"] = 64 * 1024**2
    assert "dataloader_shm_risk" in check_request(request)["issues"]


def test_probe_configuration_requires_script_placeholder(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "candidate.py").write_text("pass")
    monkeypatch.setenv("LIGHTNING_PREFLIGHT_CPU_PROBE_IMAGE", "candidate-cpu:test")
    monkeypatch.setenv("LIGHTNING_PREFLIGHT_CPU_PROBE_ARGV", '["python", "probe.py", "{epochs}"]')
    try:
        probe_from_environment(tmp_path / "candidate.py", tmp_path, "node-1")
    except ValueError as exc:
        assert "{script_path}" in str(exc)
    else:
        raise AssertionError("probe without the candidate script was accepted")


def test_probe_configuration_substitutes_exact_candidate_and_two_epochs(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "candidate.py").write_text("pass")
    monkeypatch.setenv("LIGHTNING_PREFLIGHT_CPU_PROBE_IMAGE", "candidate-cpu:test")
    monkeypatch.setenv(
        "LIGHTNING_PREFLIGHT_CPU_PROBE_ARGV",
        '["python", "probe.py", "{script_path}", "--epochs", "{epochs}", "--node", "{node_id}"]',
    )
    probe = probe_from_environment(tmp_path / "candidate.py", tmp_path, "node-1")
    assert probe.argv[-7:] == (
        "candidate-cpu:test", "probe.py", "/candidate/candidate.py",
        "--epochs", "2", "--node", "node-1",
    )


def test_opt_in_bridge_requires_isolated_cpu_image_before_scheduler_submit(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LIGHTNING_PREFLIGHT_SPECULATIVE", "1")
    monkeypatch.setenv(
        "LIGHTNING_PREFLIGHT_CPU_PROBE_ARGV",
        '["python", "probe.py", "{script_path}", "{epochs}"]',
    )
    monkeypatch.delenv("LIGHTNING_PREFLIGHT_CPU_PROBE_IMAGE", raising=False)
    source = tmp_path / "candidate.py"
    source.write_text("print('candidate')")
    submitted = []

    class Client:
        def submit(self, job):
            submitted.append(job)
            return type("Job", (), {"job_id": "unsafe"})()

        def cancel(self, job_id):
            return None

    try:
        submit_with_speculation(
            Client(), object(), source.read_text(), source, tmp_path, "unsafe",
            {"admitted": True},
        )
    except ValueError as exc:
        assert "LIGHTNING_PREFLIGHT_CPU_PROBE_IMAGE" in str(exc)
    else:
        raise AssertionError("speculation admitted an unsandboxed CPU probe")
    assert submitted == []


def test_probe_command_is_cpu_only_and_mounts_candidate_read_only(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "candidate.py"
    source.write_text("print('candidate')")
    monkeypatch.setenv("LIGHTNING_PREFLIGHT_CPU_PROBE_IMAGE", "candidate-cpu:test")
    monkeypatch.setenv(
        "LIGHTNING_PREFLIGHT_CPU_PROBE_ARGV",
        '["python", "/checker/cpu_probe.py", "{script_path}", "{working_directory}", "{epochs}"]',
    )
    command = probe_from_environment(source, tmp_path, "node-1").argv
    assert command[:3] == ("docker", "run", "--rm")
    assert ("--network", "none") == command[command.index("--network"):command.index("--network") + 2]
    assert "--read-only" in command
    assert ("--cap-drop", "ALL") == command[command.index("--cap-drop"):command.index("--cap-drop") + 2]
    assert ("--security-opt", "no-new-privileges") == command[command.index("--security-opt"):command.index("--security-opt") + 2]
    assert ("--user", "10001:10001") == command[command.index("--user"):command.index("--user") + 2]
    assert f"type=bind,src={tmp_path},dst=/candidate,readonly" in command
    assert command[-5:] == (
        "candidate-cpu:test", "/checker/cpu_probe.py", "/candidate/candidate.py", "/candidate", "2",
    )


def test_opt_in_scheduler_bridge_cancels_failed_probe(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LIGHTNING_PREFLIGHT_SPECULATIVE", "1")
    monkeypatch.setenv("LIGHTNING_PREFLIGHT_CPU_PROBE_IMAGE", "candidate-cpu:test")
    monkeypatch.setenv(
        "LIGHTNING_PREFLIGHT_CPU_PROBE_ARGV",
        json.dumps([sys.executable, "-c", "import json;print(json.dumps({'status':'failed','bug_confirmed':True,'epochs_completed':1}))", "{script_path}", "{epochs}"]),
    )
    script = tmp_path / "node.py"
    script.write_text("print('node')")
    from Preflight_checker import plugin
    monkeypatch.setattr(
        plugin, "probe_from_environment",
        lambda *_: ProbeSpec(
            (sys.executable, "-c", "import json;print(json.dumps({'status':'failed','bug_confirmed':True,'epochs_completed':1}))"),
            tmp_path,
        ),
    )

    class Client:
        cancelled = []
        submitted = []

        def submit(self, job):
            self.submitted.append(job)
            return type("Job", (), {"job_id": "job-3"})()

        def cancel(self, job_id):
            self.cancelled.append(job_id)

    client = Client()
    submitted, decision = submit_with_speculation(
        client, object(), "print('node')", script, tmp_path, "n3",
        {"admitted": True, "status": "syntax_valid", "diagnostics": []},
    )
    assert submitted.job_id == "job-3"
    assert len(client.submitted) == 1
    assert client.cancelled == ["job-3"]
    assert decision["status"] == "probe_rejected"


def test_real_two_epoch_adapter_runs_candidate_hook_twice(tmp_path: Path) -> None:
    source = tmp_path / "candidate.py"
    source.write_text(
        "def preflight_train_epoch(epoch, fixture_root):\n"
        "    assert epoch in (0, 1)\n"
        "    with open(fixture_root / 'epochs.txt', 'a') as out:\n"
        "        out.write(str(epoch))\n"
        "    return 2.0 - epoch\n"
    )
    result = SpeculativePreflight().run(
        source.read_text(), workspace=tmp_path, node_id="real-epochs",
        submit=lambda: "job-4", cancel=lambda job_id: None,
        probe=ProbeSpec(
            (sys.executable, str(Path(__file__).with_name("cpu_probe.py")), str(source), str(tmp_path), "2"),
            tmp_path,
        ),
    )
    assert result["status"] == "probe_passed"
    assert (tmp_path / "epochs.txt").read_text() == "01"
    assert result["probe"]["evidence"]["losses"] == [2.0, 1.0]


def test_read_only_fixture_error_is_inconclusive_not_candidate_bug(tmp_path: Path) -> None:
    source = tmp_path / "candidate.py"
    source.write_text(
        "def preflight_train_epoch(epoch, fixture_root):\n"
        "    raise PermissionError('read-only fixture')\n"
    )
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("cpu_probe.py")), str(source), str(tmp_path), "2"],
        capture_output=True, text=True, check=True,
    )
    evidence = json.loads(completed.stdout.strip().splitlines()[-1])
    assert evidence["status"] == "inconclusive"
    assert evidence.get("bug_confirmed") is not True


@pytest.mark.parametrize("reproduce_bug", [False, True])
def test_docker_cpu_probe_runs_two_epochs_and_cancels_only_confirmed_bug(
    tmp_path: Path, monkeypatch, reproduce_bug: bool,
) -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker is unavailable")
    if subprocess.run(
        ["docker", "image", "inspect", "lightning-preflight:local"],
        capture_output=True, check=False,
    ).returncode != 0:
        pytest.skip("Local CPU probe smoke image is unavailable")
    tmp_path.chmod(0o755)
    source = tmp_path / "candidate.py"
    source.write_text(
        "def preflight_train_epoch(epoch, fixture_root):\n"
        "    import os\n"
        "    from pathlib import Path\n"
        "    assert os.geteuid() == 10001\n"
        "    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'\n"
        "    assert os.environ['NVIDIA_VISIBLE_DEVICES'] == 'void'\n"
        "    assert Path.cwd() == Path('/scratch')\n"
        "    assert Path('/candidate/candidate.py').is_file()\n"
        + ("    if epoch == 1: raise ValueError('reproduced training bug')\n" if reproduce_bug else "")
        + "    return 2.0 - epoch\n"
    )
    monkeypatch.setenv("LIGHTNING_PREFLIGHT_SPECULATIVE", "1")
    monkeypatch.setenv("LIGHTNING_PREFLIGHT_CPU_PROBE_IMAGE", "lightning-preflight:local")
    monkeypatch.setenv(
        "LIGHTNING_PREFLIGHT_CPU_PROBE_ARGV",
        '["python", "/checker/cpu_probe.py", "{script_path}", "{working_directory}", "{epochs}"]',
    )
    submitted, cancelled = [], []

    class Client:
        def submit(self, job):
            submitted.append(job)
            return type("Job", (), {"job_id": "docker-job"})()

        def cancel(self, job_id):
            cancelled.append(job_id)

    job = object()
    _, decision = submit_with_speculation(
        Client(), job, source.read_text(), source, tmp_path, "docker-node",
        {"admitted": True},
    )
    assert submitted == [job]
    assert decision["status"] == ("probe_rejected" if reproduce_bug else "probe_passed")
    assert cancelled == (["docker-job"] if reproduce_bug else [])
    assert decision["probe"]["epochs_completed"] == (1 if reproduce_bug else 2)


def test_timed_out_docker_probe_leaves_no_running_container(tmp_path: Path) -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker is unavailable")
    if subprocess.run(
        ["docker", "image", "inspect", "lightning-preflight:local"],
        capture_output=True, check=False,
    ).returncode != 0:
        pytest.skip("Local CPU probe smoke image is unavailable")
    name = f"preflight-timeout-test-{uuid.uuid4().hex[:12]}"
    spec = ProbeSpec(
        ("docker", "run", "--rm", "--name", name, "--entrypoint", "python",
         "lightning-preflight:local", "-c", "import time; time.sleep(30)"),
        tmp_path,
        timeout_seconds=1,
    )
    try:
        assert _cpu_probe(spec)["reason"] == "timeout"
        assert subprocess.run(
            ["docker", "inspect", name], capture_output=True, check=False,
        ).returncode != 0
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def test_docker_cleanup_error_does_not_crash_scheduler_bridge(tmp_path: Path, monkeypatch) -> None:
    if shutil.which("docker") is None:
        pytest.skip("Docker is unavailable")
    if subprocess.run(
        ["docker", "image", "inspect", "lightning-preflight:local"],
        capture_output=True, check=False,
    ).returncode != 0:
        pytest.skip("Local CPU probe smoke image is unavailable")
    name = f"preflight-cleanup-test-{uuid.uuid4().hex[:12]}"
    original_run = subprocess.run

    def fail_cleanup(argv, **kwargs):
        if argv[:3] == ["docker", "rm", "-f"]:
            raise subprocess.TimeoutExpired(argv, timeout=15)
        return original_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_cleanup)
    spec = ProbeSpec(
        ("docker", "run", "--rm", "--name", name, "--entrypoint", "python",
         "lightning-preflight:local", "-c", "import time; time.sleep(30)"),
        tmp_path, timeout_seconds=1,
    )
    try:
        result = _cpu_probe(spec)
        assert result["reason"] == "timeout"
        assert result["container_cleanup"] == "failed"
    finally:
        original_run(["docker", "rm", "-f", name], capture_output=True, check=False)


def test_real_torch_training_runs_two_epochs_in_cpu_container(tmp_path: Path, monkeypatch) -> None:
    image = "downeyflyfan/downeyflyfan-default-docker:latest"
    if shutil.which("docker") is None or subprocess.run(
        ["docker", "image", "inspect", image], capture_output=True, check=False,
    ).returncode != 0:
        pytest.skip("Candidate PyTorch image is unavailable")
    tmp_path.chmod(0o755)
    source = tmp_path / "candidate.py"
    source.write_text(
        "import torch\n"
        "torch.set_num_threads(1)\n"
        "model = torch.nn.Linear(1, 1)\n"
        "with torch.no_grad():\n"
        "    model.weight.zero_()\n"
        "    model.bias.zero_()\n"
        "optimizer = torch.optim.SGD(model.parameters(), lr=0.1)\n"
        "x = torch.ones((8, 1))\n"
        "y = torch.ones((8, 1))\n"
        "def preflight_train_epoch(epoch, fixture_root):\n"
        "    assert not torch.cuda.is_available()\n"
        "    optimizer.zero_grad()\n"
        "    loss = torch.nn.functional.mse_loss(model(x), y)\n"
        "    loss.backward()\n"
        "    optimizer.step()\n"
        "    return loss.item()\n"
    )
    monkeypatch.setenv("LIGHTNING_PREFLIGHT_CPU_PROBE_IMAGE", image)
    monkeypatch.setenv(
        "LIGHTNING_PREFLIGHT_CPU_PROBE_ARGV",
        '["python", "/checker/cpu_probe.py", "{script_path}", "{working_directory}", "{epochs}"]',
    )
    cancelled = []
    result = SpeculativePreflight().run(
        source.read_text(), workspace=tmp_path, node_id="torch-two-epochs",
        submit=lambda: "torch-job", cancel=cancelled.append,
        probe=probe_from_environment(source, tmp_path, "torch-two-epochs"),
    )
    assert result["status"] == "probe_passed"
    assert result["probe"]["epochs_completed"] == 2
    assert result["probe"]["evidence"]["losses"] == pytest.approx([1.0, 0.36])
    assert cancelled == []
