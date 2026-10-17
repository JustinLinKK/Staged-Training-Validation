import json
import os
import selectors
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from model_preflight.cli import main
from model_preflight.core.results import PreflightReport
from model_preflight.reporting.json_report import validate_report

FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.mark.parametrize("fixture, status, exit_code", [
    ("valid", "FULL_PASS", 0), ("broken", "FAIL", 10),
])
def test_events_stream_static_admission_then_real_workflow_result(
    tmp_path, fixture, status, exit_code,
) -> None:
    raw = yaml.safe_load((FIXTURES / f"{fixture}.yaml").read_text())
    raw["candidate"]["root"] = str(FIXTURES / f"{fixture}_candidate")
    raw["scenarios"]["train_batch_sizes"] = [2]
    raw["execution"]["max_workers"] = 2
    path = tmp_path / "preflight.yaml"
    path.write_text(yaml.safe_dump(raw))
    report_path = tmp_path / "final.json"
    environment = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[2] / "src")}
    with subprocess.Popen(
        [sys.executable, "-m", "model_preflight.cli", "check", str(path),
         "--format", "events", "--report", str(report_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment,
    ) as process:
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                assert selector.select(timeout=15), "provisional response was not flushed"
            first = json.loads(process.stdout.readline())
            assert first["status"] == "STATIC_PASS"
            assert not first["terminal"]
            assert process.poll() is None, "workflow should still be running"
            output, stderr = process.communicate(timeout=90)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    events = [json.loads(line) for line in output.splitlines()]
    assert len(events) == 1, stderr
    final = events[0]
    assert final["status"] == status
    assert final["terminal"]
    assert final["request_id"] == first["request_id"]
    assert final["sequence"] == 2
    assert process.returncode == exit_code, stderr
    report = PreflightReport.from_dict(final["report"])
    validate_report(report)
    assert json.loads(report_path.read_text()) == final["report"]
    assert len([s for s in report.stages if s.name == "static_source"]) == 1
    assert {s.name for s in report.stages} == {
        "static_source", "hardware", "construction", "data_contract", "abstract_forward",
        "cpu_training", "validation", "memory",
    }
    if status == "FAIL":
        assert "AUT001" in {d.code for d in report.diagnostics}


def test_events_reject_partial_workflows() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["check", str(FIXTURES / "valid.yaml"), "--format", "events", "--only", "hardware"])
    assert exc.value.code == 2


def test_events_invalid_manifest_has_no_provisional_admission(tmp_path, capsys) -> None:
    assert main(["check", str(tmp_path / "missing.yaml"), "--format", "events"]) == 30
    assert capsys.readouterr().out == ""


def test_static_rejection_emits_only_terminal_event(tmp_path, capsys) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "preflight_adapter.py").write_text("def syntax error:\n")
    raw = yaml.safe_load((FIXTURES / "valid.yaml").read_text())
    raw["candidate"]["root"] = str(candidate)
    manifest = tmp_path / "preflight.yaml"
    manifest.write_text(yaml.safe_dump(raw))
    assert main(["check", str(manifest), "--format", "events"]) == 10
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(events) == 1
    assert events[0]["status"] == "FAIL"
    assert events[0]["sequence"] == 1
    assert events[0]["terminal"]
    assert not events[0]["speculative_submission_allowed"]
