from dataclasses import replace
from pathlib import Path
from threading import Event, Lock

import pytest

from model_preflight.core.enums import StageName, StageStatus
from model_preflight.core.results import StageResult
from model_preflight.engine.cache import cache_key
from model_preflight.engine.runner import PreflightRunner
from model_preflight.execution.worker_protocol import WorkerRequest
from model_preflight.manifest import load_manifest

FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.mark.parametrize("workers", [1, 2])
def test_bounded_parallel_scenarios_preserve_order(workers: int, monkeypatch) -> None:
    manifest = load_manifest(FIXTURES / "valid.yaml")
    manifest = replace(manifest, execution=replace(manifest.execution, max_workers=workers))
    runner = PreflightRunner(manifest, only={"abstract_forward"})
    first_started, second_finished = Event(), Event()
    lock = Lock()
    active = peak = 0
    completed = []

    def run(request: WorkerRequest, timeout: float) -> StageResult:
        nonlocal active, peak
        batch = request.scenario["batch_size"]
        with lock:
            active += 1
            peak = max(peak, active)
        if workers == 2:
            if batch == 2:
                first_started.set()
                assert second_finished.wait(5)
            elif batch == 4:
                assert first_started.wait(5)
        with lock:
            completed.append(batch)
            active -= 1
        if batch == 4:
            second_finished.set()
        return StageResult(
            name=request.stage, status=StageStatus.PASS, duration_seconds=10,
            evidence={"batch_size": batch},
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    report = runner.run()
    stage = report.stages[0]
    assert peak == workers
    assert sorted(completed) == list(manifest.scenarios.train_batch_sizes)
    assert [s["batch_size"] for s in stage.evidence["scenarios"]] == [2, 4, 8, 16, 32]
    if workers == 2:
        assert completed.index(4) < completed.index(2)
        assert stage.evidence["worker_seconds"] == 50
        assert stage.evidence["parallel_workers"] == 2
        assert stage.duration_seconds < stage.evidence["worker_seconds"]
    else:
        assert completed == [2, 4, 8, 16, 32]
        assert stage.duration_seconds == 50


def test_parallel_retry_preserves_scenario_attribution(monkeypatch) -> None:
    manifest = load_manifest(FIXTURES / "valid.yaml")
    manifest = replace(manifest, execution=replace(manifest.execution, max_workers=2))
    runner = PreflightRunner(manifest)
    requests = []
    lock = Lock()

    def run(request: WorkerRequest, timeout: float) -> StageResult:
        with lock:
            requests.append((request.scenario["batch_size"], request.shape_reduction_allowed))
        retry = request.scenario["batch_size"] == 1 and request.shape_reduction_allowed
        return StageResult(
            name=request.stage,
            status=StageStatus.INCONCLUSIVE if retry else StageStatus.PASS,
            duration_seconds=2,
            evidence={
                "batch_size": request.scenario["batch_size"],
                "shape_reduction": {"applied": request.shape_reduction_allowed},
            },
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    result = runner._scenarios(
        StageName.CPU_TRAINING, manifest.scenarios.cpu_scenarios(), 10, reduction_allowed=True,
    )
    assert result.status is StageStatus.PASS
    assert sorted(requests) == [(1, False), (1, True), (2, True)]
    normal, last = result.evidence["scenarios"]
    assert normal["batch_size"] == 2
    assert "reduction_attempt" not in normal
    assert last["batch_size"] == 1
    assert last["reduction_attempt"]["evidence"]["batch_size"] == 1
    assert last["shape_reduction"]["reason"] == "original_width_retry"
    assert last["duration_seconds"] == 4
    assert result.evidence["worker_seconds"] == 6


def test_parallel_settings_separate_cache_entries() -> None:
    manifest = load_manifest(FIXTURES / "valid.yaml")
    keys = {
        cache_key(PreflightRunner(replace(
            manifest,
            execution=replace(manifest.execution, max_workers=workers, cpu_threads=threads),
        ))._cache_inputs())
        for workers, threads in [(1, 1), (2, 1), (1, 2)]
    }
    assert len(keys) == 3
