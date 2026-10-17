from pathlib import Path

import pytest

from model_preflight.core.exceptions import ManifestError
from model_preflight.manifest import load_manifest

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_manifest_paths_are_normalized() -> None:
    manifest = load_manifest(FIXTURES / "valid.yaml")
    assert manifest.candidate.root.is_absolute()
    assert manifest.candidate.root.name == "valid_candidate"
    assert manifest.scenarios.train_batch_sizes == (2, 4, 8, 16, 32)
    assert manifest.policy.mode.value == "balanced"


def test_unknown_schema_version_is_manifest_error(tmp_path: Path) -> None:
    path = tmp_path / "preflight.yaml"
    path.write_text("schema_version: 99\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="schema_version"):
        load_manifest(path)


def test_missing_candidate_is_rejected_before_execution(tmp_path: Path) -> None:
    path = tmp_path / "preflight.yaml"
    path.write_text(
        """
schema_version: 1
candidate:
  id: missing
  root: ./does-not-exist
  adapter: adapter:CandidateAdapter
task:
  name: test
scenarios:
  train_batch_sizes: [1]
target:
  profile: nvidia/a10_24gb
""",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="does not exist"):
        load_manifest(path)


def test_unknown_fields_are_rejected(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    root.mkdir()
    path = tmp_path / "preflight.yaml"
    path.write_text(
        f"""
schema_version: 1
candidate:
  id: invalid
  root: {root}
  adapter: adapter:CandidateAdapter
  surprise: true
task:
  name: test
scenarios:
  train_batch_sizes: [1]
target:
  profile: nvidia/a10_24gb
""",
        encoding="utf-8",
    )
    with pytest.raises(ManifestError, match="surprise"):
        load_manifest(path)



@pytest.mark.parametrize("settings", [
    {"shape_reduction": {"mode": "unknown"}},
    {"shape_reduction": {"width_factor": 0}},
    {"shape_reduction": {"width_factor": 1}},
    {"shape_reduction": {"minimum_width": 15}},
    {"shape_reduction": {"surprise": True}},
    {"seed": -1},
])
def test_invalid_reduction_settings_rejected(tmp_path: Path, settings: dict) -> None:
    import yaml

    raw = yaml.safe_load((FIXTURES / "valid.yaml").read_text())
    raw["candidate"]["root"] = str(FIXTURES / "valid_candidate")
    raw["execution"].update(settings)
    path = tmp_path / "preflight.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_reduction_defaults_and_seed_round_trip() -> None:
    from dataclasses import replace

    manifest = load_manifest(FIXTURES / "valid.yaml")
    assert manifest.execution.shape_reduction.mode == "off"
    assert "seed" not in manifest.to_dict()["execution"]
    seeded = replace(manifest, execution=replace(manifest.execution, seed=123))
    assert seeded.to_dict()["execution"]["seed"] == 123
