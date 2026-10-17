"""Catch a full-test-array plus one-batch prediction shape mismatch."""

from Preflight_checker.preflight import check_source


BUGGY_BATCH_ACCUMULATION = """
import numpy as np
from torch.utils.data import DataLoader

BATCH_SIZE = 32
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
test_preds_accum = np.zeros(len(test_df), dtype=np.float32)
for images, meta, _ in test_loader:
    logits = model(images, meta)
    logits_flip = model(images, meta)
    test_preds_accum += ((logits + logits_flip) / 2.0).float().cpu().numpy()
"""


def test_rejects_unindexed_batch_predictions_added_to_full_test_array():
    report = check_source(BUGGY_BATCH_ACCUMULATION, shared_memory_bytes=0)

    assert report["admitted"] is False
    assert report["assurance"] == "bounded_static_risk"
    assert "unindexed_batch_prediction_accumulation" in report["issues"]
    assert any(
        item["code"] == "unindexed_batch_prediction_accumulation" and item["line"] == 11
        for item in report["diagnostics"]
    )


def test_accepts_batch_predictions_collected_then_concatenated():
    source = BUGGY_BATCH_ACCUMULATION.replace(
        "test_preds_accum = np.zeros(len(test_df), dtype=np.float32)",
        "test_preds_accum = []",
    ).replace(
        "test_preds_accum += ((logits + logits_flip) / 2.0).float().cpu().numpy()",
        "test_preds_accum.append(((logits + logits_flip) / 2.0).float().cpu().numpy())\n"
        "test_preds = np.concatenate(test_preds_accum)",
    )

    report = check_source(source, shared_memory_bytes=0)

    assert report["admitted"] is True
    assert report["issues"] == []


def test_accepts_indexed_batch_accumulation():
    source = BUGGY_BATCH_ACCUMULATION.replace(
        "for images, meta, _ in test_loader:",
        "for batch_index, (images, meta, _) in enumerate(test_loader):",
    ).replace(
        "test_preds_accum += ((logits + logits_flip) / 2.0).float().cpu().numpy()",
        "test_preds_accum[batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE] "
        "+= ((logits + logits_flip) / 2.0).float().cpu().numpy()",
    )

    report = check_source(source, shared_memory_bytes=0)

    assert report["admitted"] is True
    assert report["issues"] == []
