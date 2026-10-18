"""Catch incompatible shared feature caches before a candidate executes."""

import pickle

from Preflight_checker.preflight import check_source


SOURCE = '''\
import os
import pickle
WORK_DIR = "./working"
def build_and_save_features():
    with open(os.path.join(WORK_DIR, "feature_meta.pkl"), "wb") as out:
        pickle.dump({"cont_features": ["brightness"]}, out)
def run_training():
    with open(os.path.join(WORK_DIR, "feature_meta.pkl"), "rb") as source:
        feature_meta = pickle.load(source)
    return feature_meta["cont_features"]
if __name__ == "__main__":
    feat_meta_path = os.path.join(WORK_DIR, "feature_meta.pkl")
    if os.path.exists(feat_meta_path):
        print("cached features found; skipping build")
    else:
        build_and_save_features()
    run_training()
'''


def test_existing_cache_missing_required_metadata_key_is_rejected(tmp_path):
    working = tmp_path / "working"
    working.mkdir()
    (working / "feature_meta.pkl").write_bytes(
        pickle.dumps({"meta_feats": ["brightness"]})
    )

    report = check_source(
        SOURCE, working_directory=tmp_path, shared_memory_bytes=1 << 40
    )

    assert report["admitted"] is False
    assert "cached_metadata_missing_key" in report["issues"]
    assert any("cont_features" in item["message"] for item in report["diagnostics"])


def test_existing_cache_with_required_metadata_key_is_admitted(tmp_path):
    working = tmp_path / "working"
    working.mkdir()
    (working / "feature_meta.pkl").write_bytes(
        pickle.dumps({"cont_features": ["brightness"]})
    )

    report = check_source(
        SOURCE, working_directory=tmp_path, shared_memory_bytes=1 << 40
    )

    assert report["admitted"] is True

def test_stale_cache_is_not_rejected_when_source_rebuilds_unconditionally(tmp_path):
    working = tmp_path / "working"
    working.mkdir()
    (working / "feature_meta.pkl").write_bytes(
        pickle.dumps({"meta_feats": ["brightness"]})
    )
    source = SOURCE.replace(
        'if os.path.exists(feat_meta_path):\n        print("cached features found; skipping build")\n    else:\n        build_and_save_features()',
        'build_and_save_features()',
    )

    report = check_source(
        source, working_directory=tmp_path, shared_memory_bytes=1 << 40
    )

    assert report["admitted"] is True
