"""Reject unconditional imports that the candidate interpreter cannot resolve."""

from Preflight_checker.preflight import check_source


MISSING = "preflight_test_missing_package_8367b4"


def test_rejects_unavailable_top_level_package_before_training():
    report = check_source(f"import {MISSING} as package\n", shared_memory_bytes=0)

    assert report["admitted"] is False
    assert "unavailable_import" in report["issues"]
    assert report["diagnostics"][0]["line"] == 1
    assert MISSING in report["diagnostics"][0]["message"]


def test_optional_import_with_importerror_fallback_is_admitted():
    source = (
        f"try:\n    import {MISSING} as package\n"
        "except ImportError:\n    package = None\n"
    )

    report = check_source(source, shared_memory_bytes=0)

    assert report["admitted"] is True


def test_available_standard_library_import_is_admitted():
    report = check_source("import json\n", shared_memory_bytes=0)

    assert report["admitted"] is True
