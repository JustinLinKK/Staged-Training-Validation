"""The portable preflight protocol must not silently admit untested candidates."""

import json
import sys
from pathlib import Path

from Preflight_checker.plugin import ProbeSpec, SpeculativePreflight, check_request, probe_from_environment, submit_with_speculation


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
    monkeypatch.setenv("LIGHTNING_PREFLIGHT_CPU_PROBE_ARGV", '["python", "probe.py", "{epochs}"]')
    try:
        probe_from_environment(tmp_path / "candidate.py", tmp_path, "node-1")
    except ValueError as exc:
        assert "{script_path}" in str(exc)
    else:
        raise AssertionError("probe without the candidate script was accepted")


def test_probe_configuration_substitutes_exact_candidate_and_two_epochs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(
        "LIGHTNING_PREFLIGHT_CPU_PROBE_ARGV",
        '["python", "probe.py", "{script_path}", "--epochs", "{epochs}", "--node", "{node_id}"]',
    )
    probe = probe_from_environment(tmp_path / "candidate.py", tmp_path, "node-1")
    assert probe.argv == (
        "python", "probe.py", str(tmp_path / "candidate.py"),
        "--epochs", "2", "--node", "node-1",
    )


def test_opt_in_scheduler_bridge_cancels_failed_probe(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LIGHTNING_PREFLIGHT_SPECULATIVE", "1")
    monkeypatch.setenv(
        "LIGHTNING_PREFLIGHT_CPU_PROBE_ARGV",
        json.dumps([sys.executable, "-c", "import json;print(json.dumps({'status':'failed','bug_confirmed':True,'epochs_completed':1}))", "{script_path}", "{epochs}"]),
    )
    script = tmp_path / "node.py"
    script.write_text("print('node')")

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
