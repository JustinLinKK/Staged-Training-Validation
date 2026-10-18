"""Regression for test IDs consumed as numeric prediction targets."""

from textwrap import indent

from Preflight_checker.preflight import check_source


SOURCE = """
class PetDataset:
    def __init__(self, has_target=True):
        self.has_target = has_target
        self.ids = ["pet-1"]
        self.targets = [0.42]

    def __getitem__(self, idx):
        if self.has_target:
            return (0, 0, self.targets[idx])
        return (0, 0, self.ids[idx])

test_ds = PetDataset(False)
test_loader = DataLoader(test_ds, batch_size=32)

def predict(model, loader, return_targets=False):
    trues = []
    for image, meta, extra in loader:
        if return_targets:
            trues.append((extra * 100.0).cpu())

def predict_with_state(model, state, loader, return_targets=True):
    return predict(model, loader, return_targets=return_targets)

test_predictions = predict_with_state(model, state, test_loader)
"""


def test_rejects_string_ids_consumed_as_numeric_targets_through_defaulted_wrapper():
    report = check_source(SOURCE, shared_memory_bytes=0)

    assert report["admitted"] is False
    assert "test_ids_used_as_numeric_targets" in report["issues"]
    assert any(
        diagnostic["code"] == "test_ids_used_as_numeric_targets"
        and diagnostic["line"] == 25
        for diagnostic in report["diagnostics"]
    )


def test_accepts_explicit_prediction_without_targets_for_test_ids():
    source = SOURCE.replace(
        "predict_with_state(model, state, test_loader)",
        "predict_with_state(model, state, test_loader, return_targets=False)",
    )

    report = check_source(source, shared_memory_bytes=0)

    assert report["admitted"] is True
    assert report["issues"] == []


def test_accepts_numeric_targets_when_target_collection_is_enabled():
    source = SOURCE.replace("test_ds = PetDataset(False)", "test_ds = PetDataset(True)")

    report = check_source(source, shared_memory_bytes=0)

    assert report["admitted"] is True
    assert report["issues"] == []


def test_rejects_same_mismatch_inside_fold_loop():
    declarations, helpers = SOURCE.split("def predict(model", 1)
    declarations = declarations.replace(
        "test_ds = PetDataset(False)\ntest_loader = DataLoader(test_ds, batch_size=32)\n",
        "",
    )
    helpers = "def predict(model" + helpers
    helpers = helpers.replace(
        "test_predictions = predict_with_state(model, state, test_loader)",
        "for fold in range(5):\n"
        + indent(
            "test_ds = PetDataset(False)\n"
            "test_loader = DataLoader(test_ds, batch_size=32)\n"
            "test_predictions = predict_with_state(model, state, test_loader)",
            "    ",
        ),
    )

    report = check_source(declarations + helpers, shared_memory_bytes=0)

    assert "test_ids_used_as_numeric_targets" in report["issues"]
