import json
from concurrent.futures import TimeoutError
from pathlib import Path
from threading import Event

import pytest
import yaml

from model_preflight import submit
from model_preflight.core.enums import OverallStatus, StageStatus
from model_preflight.core.results import PreflightReport
from model_preflight.engine.runner import PreflightRunner

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _report(runner, static, status=OverallStatus.PASS):
    return PreflightReport(
        candidate_id=runner.manifest.candidate.id,
        candidate_hash=runner.candidate_hash,
        checker_version="test",
        overall_status=status,
        gpu_submission_recommended=status is OverallStatus.PASS,
        normalized_manifest=runner.manifest.to_dict(),
        stages=[static],
        diagnostics=[],
    )


@pytest.mark.parametrize("outcome", [*OverallStatus, "exception"])
def test_static_pass_returns_before_workflow_finishes(monkeypatch, outcome) -> None:
    started, release = Event(), Event()
    calls = []

    def run(self, *, static_result=None):
        calls.append(static_result)
        started.set()
        assert release.wait(5)
        if outcome == "exception":
            raise RuntimeError("controller failed")
        return _report(self, static_result, outcome)

    monkeypatch.setattr(PreflightRunner, "run", run)
    submission = submit(FIXTURES / "valid.yaml")
    try:
        assert started.wait(5)
        initial = submission.initial
        assert initial.status == "STATIC_PASS"
        assert not initial.terminal
        assert initial.report is None
        assert initial.to_dict()["gpu_check_required"]
        assert submission.poll() is initial
        with pytest.raises(TimeoutError):
            submission.completion.result(timeout=0.01)
    finally:
        release.set()
    final = submission.completion.result(timeout=5)
    expected = "INTERNAL_ERROR" if outcome == "exception" else outcome.value
    assert final.status == ("FULL_PASS" if expected == "PASS" else expected)
    assert final.terminal
    assert final.request_id == initial.request_id
    assert final.candidate_hash == initial.candidate_hash
    assert final.sequence == 2
    assert not final.to_dict()["speculative_submission_allowed"]
    assert submission.poll() is final
    assert len(calls) == 1 and calls[0].status is StageStatus.PASS
    if outcome == "exception":
        assert final.error["code"] == "CONTROLLER_ERROR"
        assert final.report is None


def _manifest(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    source = candidate / "preflight_adapter.py"
    source.write_text((FIXTURES / "valid_candidate/preflight_adapter.py").read_text())
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "data.txt").write_text("initial")
    raw = yaml.safe_load((FIXTURES / "valid.yaml").read_text())
    raw["candidate"]["root"] = str(candidate)
    raw["task"]["fixture_root"] = str(fixture)
    raw["execution"]["cache"] = True
    path = tmp_path / "preflight.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path, source, fixture / "data.txt"


@pytest.mark.parametrize("input_index", [0, 1, 2])
def test_changed_inputs_cannot_receive_full_pass(tmp_path, monkeypatch, input_index) -> None:
    paths = _manifest(tmp_path)

    def run(self, *, static_result=None):
        assert not self.use_cache
        with paths[input_index].open("a") as stream:
            stream.write("\n# changed\n")
        return _report(self, static_result)

    monkeypatch.setattr(PreflightRunner, "run", run)
    final = submit(paths[0]).completion.result(timeout=5)
    assert final.status == "INCONCLUSIVE"
    assert final.error["code"] == "INPUT_CHANGED"
    assert final.report is None


def test_static_rejection_is_terminal_and_never_starts_workers(tmp_path, monkeypatch) -> None:
    manifest, source, _ = _manifest(tmp_path)
    source.write_text("def broken syntax(\n")

    def unexpected(*args, **kwargs):
        pytest.fail("static rejection must not start a worker")

    monkeypatch.setattr(PreflightRunner, "_worker", unexpected)
    submission = submit(manifest)
    assert submission.initial.status == "FAIL"
    assert submission.initial.terminal
    assert submission.initial.sequence == 1
    assert submission.completion.done()
    assert submission.poll() is submission.initial
    assert [s.name for s in submission.initial.report.stages] == ["static_source"]


def test_inputs_changed_during_static_check_prevent_workflow(tmp_path, monkeypatch) -> None:
    manifest, source, _ = _manifest(tmp_path)
    run_static = PreflightRunner.run_static

    def changing_static(self):
        result = run_static(self)
        source.write_text("# replaced after static check\n")
        return result

    def unexpected(*args, **kwargs):
        pytest.fail("changed inputs must not start the workflow")

    monkeypatch.setattr(PreflightRunner, "run_static", changing_static)
    monkeypatch.setattr(PreflightRunner, "run", unexpected)
    final = submit(manifest).completion.result(timeout=5)
    assert final.status == "INCONCLUSIVE"
    assert final.error["code"] == "INPUT_CHANGED"


def test_concurrent_submissions_have_independent_identities(monkeypatch) -> None:
    def run(self, *, static_result=None):
        return _report(self, static_result)

    monkeypatch.setattr(PreflightRunner, "run", run)
    submissions = [submit(FIXTURES / "valid.yaml") for _ in range(2)]
    updates = [s.completion.result(timeout=5) for s in submissions]
    assert updates[0].request_id != updates[1].request_id
    assert updates[0].candidate_hash == updates[1].candidate_hash
    assert updates[0].manifest_hash == updates[1].manifest_hash
    assert all(json.loads(json.dumps(u.to_dict()))["status"] == "FULL_PASS" for u in updates)
