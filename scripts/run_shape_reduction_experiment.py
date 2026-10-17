#!/usr/bin/env python3
"""Freeze, execute, and summarize paired preflight reduction experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from shape_reduction_corpus import archive_cases, cases, manifest, scale_cases

REPO = Path(__file__).resolve().parents[1]
QUALITY_SEEDS = [11, 23, 37]
TIMING_SEEDS = [101, 103, 107, 109, 113]
CONTROLLER = """
import json, sys, traceback
from model_preflight import check
try:
    report = check(sys.argv[1], use_cache=False).to_dict()
except Exception as exc:
    report = {"overall_status":"INTERNAL_ERROR", "diagnostics":[], "stages":[],
              "controller_error":type(exc).__name__ + ": " + str(exc),
              "trace":traceback.format_exc()}
with open(sys.argv[2], "w") as stream:
    json.dump(report, stream, indent=2, sort_keys=True)
"""


def code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(
        p for p in (REPO / "src").rglob("*") if p.suffix in {".py", ".json", ".yaml"}
    ):
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def freeze(
    output: Path, archive: Path | None, scale_study: bool = False, auto_only: bool = False,
    selected: set[str] | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=False)
    corpus = scale_cases() if scale_study else cases()
    exclusions: list[dict[str, str]] = []
    if archive is not None:
        historical, exclusions = archive_cases(archive)
        corpus.extend(historical)
    if selected is not None:
        unknown = selected - {case["name"] for case in corpus}
        if unknown:
            raise SystemExit(f"Unknown corpus cases: {sorted(unknown)}")
        corpus = [case for case in corpus if case["name"] in selected]
    available = []
    for case in corpus:
        missing = [name for name in case.get("requires", [])
                   if importlib.util.find_spec(name) is None]
        if missing:
            exclusions.append({"case": case["name"], "reason": "missing_dependencies:"
                               + ",".join(missing)})
        else:
            available.append(case)
    corpus = available
    for case in corpus:
        candidate = output / "corpus" / case["name"] / "candidate"
        candidate.mkdir(parents=True)
        (candidate / "candidate.py").write_text(case.pop("source"))
        case["candidate_sha256"] = hashlib.sha256(
            (candidate / "candidate.py").read_bytes()
        ).hexdigest()
        configured = case.pop("manifest", None) or manifest(case, candidate)
        configured["candidate"]["root"] = str(candidate)
        configured["candidate"]["id"] = case["name"]
        configured["execution"]["cache"] = False
        profile = Path(configured["target"]["profile"])
        if profile.is_file():
            profile_copy = candidate.parent / "profile.yaml"
            shutil.copyfile(profile, profile_copy)
            configured["target"]["profile"] = str(profile_copy)
        if case.get("evidence_path"):
            shutil.copyfile(case["evidence_path"], candidate.parent / "historical_report.json")
        case["manifest"] = configured
    implementation = output / "implementation"
    shutil.copytree(
        REPO / "src" / "model_preflight",
        implementation / "src" / "model_preflight",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        REPO / "scripts",
        implementation / "scripts",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    if not auto_only:
        baseline = output / "baseline"
        baseline.mkdir()
        archive_path = output / "baseline.tar"
        with archive_path.open("wb") as stream:
            subprocess.run(["git", "archive", "HEAD", "src"], cwd=REPO, stdout=stream, check=True)
        with tarfile.open(archive_path) as stream:
            stream.extractall(baseline, filter="data")
        archive_path.unlink()
    frozen = {
        "corpus_kind": "scale_study" if scale_study else "standard",
        "auto_only": auto_only,
        "baseline_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "implementation_sha256": code_hash(),
        "cases": corpus,
        "exclusions": exclusions,
        "quality_seeds": QUALITY_SEEDS,
        "timing_seeds": TIMING_SEEDS,
        "thresholds": {"median_speedup": 1.25, "maximum_p95_ratio": 1.10},
        "python": sys.version,
        "worker_python": sys.executable,
        "dependency_versions": {
            name: importlib.metadata.version(name)
            for name in {"torch"} | {d for case in corpus for d in case.get("requires", [])}
        },
        "cpu_count": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0)),
    }
    (output / "corpus.json").write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    (output / "implementation.patch").write_text(
        subprocess.check_output(["git", "diff", "--", "src"], cwd=REPO, text=True)
    )
    print(
        json.dumps({"frozen_cases": len(corpus), "exclusions": exclusions, "output": str(output)}),
        flush=True,
    )


def execute(output: Path, case: dict[str, Any], phase: str, arm: str, seed: int) -> None:
    trial = output / "raw" / phase / case["name"] / f"{arm}-{seed}"
    if (trial / "measurement.json").is_file():
        return
    trial.mkdir(parents=True, exist_ok=True)
    configured = json.loads(json.dumps(case["manifest"]))
    configured["execution"].pop("shape_reduction", None)
    configured["execution"].pop("seed", None)
    if arm != "v1":
        configured["execution"]["seed"] = seed
        configured["execution"]["shape_reduction"] = {
            "mode": arm,
            "width_factor": 0.5,
            "minimum_width": 16,
        }
    config_path = trial / "preflight.yaml"
    config_path.write_text(json.dumps(configured, indent=2))
    report_path = trial / "report.json"
    source = output / ("baseline" if arm == "v1" else "implementation") / "src"
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(source) + ":/home/justin/MLEvolve",
    }
    load_start = os.getloadavg()
    started = time.monotonic()
    with (trial / "controller.log").open("w") as stream:
        process = subprocess.run(
            [sys.executable, "-c", CONTROLLER, str(config_path), str(report_path)],
            cwd=trial,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    duration = time.monotonic() - started
    if not report_path.is_file():
        report_path.write_text(
            json.dumps(
                {
                    "overall_status": "INTERNAL_ERROR",
                    "stages": [],
                    "diagnostics": [],
                    "controller_exit": process.returncode,
                }
            )
        )
    report = json.loads(report_path.read_text())
    stages = report.get("stages", [])
    evidence = [
        item
        for s in stages
        for item in s.get("evidence", {}).get("scenarios", [s.get("evidence", {})])
    ]
    evidence += [
        e["reduction_attempt"]["evidence"]
        for e in list(evidence)
        if e.get("reduction_attempt", {}).get("evidence")
    ]
    reduction = [item["shape_reduction"] for item in evidence if "shape_reduction" in item]
    record = {
        "case": case["name"],
        "group": case["group"],
        "phase": phase,
        "arm": arm,
        "seed": seed,
        "status": report["overall_status"],
        "codes": sorted({d["code"] for d in report["diagnostics"]}),
        "confirmed_codes": sorted(
            {
                d["code"]
                for d in report["diagnostics"]
                if d["classification"] == "confirmed_candidate_failure"
            }
        ),
        "duration_seconds": duration,
        "cpu_stage_seconds": sum(
            s["duration_seconds"] for s in stages if s["name"] in {"cpu_training", "validation"}
        ),
        "workflow_seconds": sum(e.get("workflow_seconds", 0) for e in evidence),
        "reduction_seconds": sum(r.get("duration_seconds", 0) for r in reduction),
        "peak_worker_rss_bytes": max(
            (e.get("peak_worker_rss_bytes", 0) for e in evidence), default=0
        ),
        "reduced_scenarios": sum(bool(r.get("applied")) for r in reduction),
        "eligible_scenarios": sum("duration_seconds" in r for r in reduction),
        "original_width_retries": sum("reduction_attempt" in e for e in evidence),
        "reduction_reasons": [r.get("reason") for r in reduction if not r.get("applied")],
        "load_start": load_start,
        "load_end": os.getloadavg(),
        "report": str(report_path),
    }
    (trial / "measurement.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        f"{phase} {case['name']} {arm} seed={seed}: {record['status']} "
        f"{duration:.2f}s reduced={record['reduced_scenarios']}",
        flush=True,
    )


def run(output: Path, phases: list[str], selected: set[str] | None) -> None:
    frozen = json.loads((output / "corpus.json").read_text())
    if frozen["implementation_sha256"] != code_hash():
        raise SystemExit("Checker changed since freeze; use a new experiment directory.")
    corpus = [c for c in frozen["cases"] if selected is None or c["name"] in selected]
    if frozen.get("auto_only"):
        # This audit never runs the reference or reduction-off arm, even with CLI defaults.
        phases = ["quality"]
    for case in corpus:
        candidate = Path(case["manifest"]["candidate"]["root"]) / "candidate.py"
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != case["candidate_sha256"]:
            raise SystemExit(f"Frozen candidate changed: {case['name']}")
    if "reference" in phases:
        for case in corpus:
            execute(output, case, "reference", "v1", 0)
    if "quality" in phases:
        for index, case in enumerate(corpus):
            for repeat, seed in enumerate(frozen["quality_seeds"]):
                arms = ["off", "auto"] if (index + repeat) % 2 == 0 else ["auto", "off"]
                if frozen.get("auto_only"):
                    arms = ["auto"]
                for arm in arms:
                    execute(output, case, "quality", arm, seed)
    if "timing" in phases:
        for index, case in enumerate(c for c in corpus if c.get("performance")):
            for repeat, seed in enumerate(frozen["timing_seeds"]):
                arms = ["off", "auto"] if (index + repeat) % 2 == 0 else ["auto", "off"]
                for arm in arms:
                    execute(output, case, "timing", arm, seed)


def percentile(values: list[float], fraction: float) -> float | None:
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def interval(values: list[float]) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(719)
    samples = [statistics.median(rng.choices(values, k=len(values))) for _ in range(2000)]
    return [float(percentile(samples, 0.025) or 0), float(percentile(samples, 0.975) or 0)]


def summarize(output: Path, plots: bool) -> dict[str, Any]:
    frozen = json.loads((output / "corpus.json").read_text())
    records = [
        json.loads(p.read_text()) for p in sorted((output / "raw").rglob("measurement.json"))
    ]
    for record in records:
        report = json.loads(Path(record["report"]).read_text())
        record["checker_errors"] = sorted({
            f"{d['stage']}:{d['code']}" for d in report["diagnostics"]
            if d["classification"] == "checker_error"
        })
    index = {(r["case"], r["phase"], r["arm"], r["seed"]): r for r in records}
    regressions, false_rejections, baseline_misses, reference_changes, diagnostic_changes = (
        [],
        [],
        [],
        [],
        [],
    )
    new_errors = []
    performance = []
    for case in frozen["cases"]:
        name = case["name"]
        reference = index.get((name, "reference", "v1", 0))
        ratios, cpu_ratios, workflow_ratios = [], [], []
        for phase, seeds in [
            ("quality", frozen["quality_seeds"]),
            ("timing", frozen["timing_seeds"]),
        ]:
            for seed in seeds:
                off, auto = (
                    index.get((name, phase, "off", seed)),
                    index.get((name, phase, "auto", seed)),
                )
                if not off or not auto:
                    continue
                if phase == "timing":
                    ratios.append(off["duration_seconds"] / auto["duration_seconds"])
                    cpu_ratios.append(off["cpu_stage_seconds"] / auto["cpu_stage_seconds"])
                    if auto["workflow_seconds"]:
                        workflow_ratios.append(off["workflow_seconds"] / auto["workflow_seconds"])
                    continue
                label = {"case": name, "seed": seed}
                if (
                    auto["status"] == "INTERNAL_ERROR" and off["status"] != "INTERNAL_ERROR"
                ) or set(auto["checker_errors"]) - set(off["checker_errors"]):
                    new_errors.append(label)
                if (
                    reference
                    and seed == frozen["quality_seeds"][0]
                    and (reference["status"], reference["confirmed_codes"])
                    != (off["status"], off["confirmed_codes"])
                ):
                    reference_changes.append(label)
                if (off["status"], off["confirmed_codes"]) != (
                    auto["status"],
                    auto["confirmed_codes"],
                ):
                    diagnostic_changes.append(
                        {
                            **label,
                            "off": off["status"],
                            "auto": auto["status"],
                            "off_codes": off["confirmed_codes"],
                            "auto_codes": auto["confirmed_codes"],
                        }
                    )
                if case["expected_status"] == "FAIL":
                    caught_off = off["status"] == "FAIL" and set(case["expected_codes"]).issubset(
                        off["confirmed_codes"]
                    )
                    caught_auto = auto["status"] == "FAIL" and set(case["expected_codes"]).issubset(
                        auto["confirmed_codes"]
                    )
                    if not caught_off:
                        baseline_misses.append(label)
                    elif not caught_auto:
                        regressions.append(label)
                elif (
                    case["expected_status"] is not None
                    and auto["status"] == "FAIL"
                    and off["status"] != "FAIL"
                ):
                    false_rejections.append(label)
        if ratios:
            performance.append(
                {
                    "case": name,
                    "pairs": len(ratios),
                    "median_speedup": statistics.median(ratios),
                    "speedup_ci95": interval(ratios),
                    "cpu_stage_speedup": statistics.median(cpu_ratios),
                    "cpu_stage_speedup_ci95": interval(cpu_ratios),
                    "workflow_speedup": statistics.median(workflow_ratios)
                    if workflow_ratios
                    else None,
                }
            )
    quality = [r for r in records if r["phase"] == "quality"]
    groups: dict[str, Any] = {}
    for group, rows in [
        ("synthetic", [r for r in quality if r["group"] != "historical"]),
        ("historical", [r for r in quality if r["group"] == "historical"]),
        ("all", quality),
    ]:
        groups[group] = {}
        for arm in ("off", "auto"):
            selected = [r for r in rows if r["arm"] == arm]
            groups[group][arm] = {
                "runs": len(selected),
                "median_seconds": statistics.median([r["duration_seconds"] for r in selected])
                if selected
                else None,
                "p95_seconds": percentile([r["duration_seconds"] for r in selected], 0.95),
                "inconclusive": sum(r["status"] == "INCONCLUSIVE" for r in selected),
                "internal_errors": sum(r["status"] == "INTERNAL_ERROR" for r in selected),
                "runs_with_checker_error_diagnostics": sum(
                    bool(r["checker_errors"]) for r in selected
                ),
                "runs_with_reduction": sum(r["reduced_scenarios"] > 0 for r in selected),
                "eligible_scenarios": sum(r.get("eligible_scenarios", 0) for r in selected),
                "reduced_scenarios": sum(r["reduced_scenarios"] for r in selected),
                "original_width_retries": sum(r["original_width_retries"] for r in selected),
                "median_peak_worker_rss_bytes": statistics.median(
                    [r["peak_worker_rss_bytes"] for r in selected]
                )
                if selected
                else None,
            }
            labels = {case["name"]: case for case in frozen["cases"]}
            positives = [r for r in selected if labels[r["case"]]["expected_status"] == "FAIL"]
            negatives = [
                r
                for r in selected
                if labels[r["case"]]["expected_status"] in {"PASS", "INCONCLUSIVE"}
            ]
            caught = sum(
                r["status"] == "FAIL"
                and set(labels[r["case"]]["expected_codes"]).issubset(r["confirmed_codes"])
                for r in positives
            )
            false_positives = sum(r["status"] == "FAIL" for r in negatives)
            groups[group][arm].update(
                {
                    "labeled_defect_runs": len(positives),
                    "detected_defect_runs": caught,
                    "recall": caught / len(positives) if positives else None,
                    "labeled_nondefect_runs": len(negatives),
                    "false_hard_rejections": false_positives,
                    "false_hard_rejection_rate": false_positives / len(negatives)
                    if negatives
                    else None,
                }
            )
    speedups = [p["median_speedup"] for p in performance]
    overall_speedup = statistics.median(speedups) if speedups else None
    off_p95, auto_p95 = groups["all"]["off"]["p95_seconds"], groups["all"]["auto"]["p95_seconds"]
    p95_ratio = auto_p95 / off_p95 if off_p95 and auto_p95 else None
    quality_complete = (
        len(quality) == len(frozen["cases"]) * 6
        and len([r for r in records if r["phase"] == "reference"]) == len(frozen["cases"])
    )
    timing_complete = len([r for r in records if r["phase"] == "timing"]) == (
        sum(bool(c.get("performance")) for c in frozen["cases"])
        * len(frozen["timing_seeds"]) * 2
    )
    complete = timing_complete and (
        frozen.get("corpus_kind") == "scale_study" or quality_complete
    )
    applied = all(
        any(
            r["case"] == c["name"] and r["arm"] == "auto" and r["reduced_scenarios"]
            for r in quality + [r for r in records if r["phase"] == "timing"]
        )
        for c in frozen["cases"]
        if c.get("expect_reduction")
    )
    summary = {
        "complete": complete,
        "corpus_kind": frozen.get("corpus_kind", "standard"),
        "reference_commit": frozen["baseline_commit"],
        "implementation_sha256": frozen["implementation_sha256"],
        "records": len(records),
        "groups": groups,
        "detection_regressions": regressions,
        "false_rejections": false_rejections,
        "baseline_label_misses": baseline_misses,
        "v1_off_differences": reference_changes,
        "diagnostic_changes": diagnostic_changes,
        "new_checker_errors": new_errors,
        "performance": performance,
        "median_end_to_end_speedup": overall_speedup,
        "speedup_ci95": interval(speedups),
        "aggregate_p95_ratio": p95_ratio,
        "supported_workloads_reduced": applied,
        "quality_gate": complete and quality_complete
        and not regressions
        and not false_rejections
        and not new_errors
        and not reference_changes,
        "performance_gate": complete
        and applied
        and overall_speedup is not None
        and overall_speedup >= 1.25
        and p95_ratio is not None
        and p95_ratio <= 1.10,
        "exclusions": frozen["exclusions"],
    }
    if frozen.get("corpus_kind") == "scale_study":
        # Timing-only evidence cannot re-evaluate the original replay acceptance gates.
        summary["quality_gate"] = None
        summary["performance_gate"] = None
    if frozen.get("auto_only"):
        expected = {
            (c["name"], "quality", "auto", seed)
            for c in frozen["cases"] for seed in frozen["quality_seeds"]
        }
        mismatches = [
            {"case": r["case"], "seed": r["seed"], "status": r["status"],
             "confirmed_codes": r["confirmed_codes"],
             "expected_status": labels[r["case"]]["expected_status"],
             "expected_codes": labels[r["case"]]["expected_codes"]}
            for r in quality
            if labels[r["case"]]["expected_status"] is not None and (
                r["status"] != labels[r["case"]]["expected_status"]
                or not set(labels[r["case"]]["expected_codes"]).issubset(r["confirmed_codes"])
            )
        ]
        fallback_violations = [
            {"case": r["case"], "seed": r["seed"], "reduced_scenarios": r["reduced_scenarios"]}
            for r in quality
            if labels[r["case"]].get("expect_fallback") and r["reduced_scenarios"]
        ]
        summary.update({
            "auto_only": True, "complete": set(index) == expected,
            "label_mismatches": mismatches,
            "fallback_violations": fallback_violations,
            "label_gate": set(index) == expected and not mismatches
            and not fallback_violations
            and not groups["all"]["auto"]["internal_errors"]
            and not groups["all"]["auto"]["runs_with_checker_error_diagnostics"]
            and applied,
            "quality_gate": None, "performance_gate": None,
        })
    # Keep category and individual-case costs visible, including fallback overhead.
    breakdown = []
    for scope, values in (
        ("category", sorted({r["group"] for r in records})),
        ("case", [c["name"] for c in frozen["cases"]]),
    ):
        for value in values:
            for phase in ("quality", "timing"):
                for arm in ("off", "auto"):
                    rows = [
                        r for r in records
                        if r["phase"] == phase and r["arm"] == arm
                        and r["group" if scope == "category" else "case"] == value
                    ]
                    if not rows:
                        continue
                    labeled = [
                        r for r in rows if labels[r["case"]]["expected_status"] == "FAIL"
                    ]
                    detected = sum(
                        r["status"] == "FAIL"
                        and set(labels[r["case"]]["expected_codes"]).issubset(r["confirmed_codes"])
                        for r in labeled
                    )
                    entry = {
                        "scope": scope, "name": value, "phase": phase, "arm": arm,
                        "runs": len(rows), "statuses": dict(Counter(r["status"] for r in rows)),
                        "labeled_defect_runs": len(labeled), "detected_defect_runs": detected,
                        "recall": detected / len(labeled) if labeled else None,
                        "eligible_scenarios": sum(r["eligible_scenarios"] for r in rows),
                        "applied_scenarios": sum(r["reduced_scenarios"] for r in rows),
                        "original_width_retries": sum(r["original_width_retries"] for r in rows),
                        "fallback_reasons": dict(Counter(
                            reason for r in rows for reason in r["reduction_reasons"]
                        )),
                    }
                    for metric in (
                        "duration_seconds", "cpu_stage_seconds", "workflow_seconds",
                        "reduction_seconds", "peak_worker_rss_bytes",
                    ):
                        samples = [r[metric] for r in rows]
                        entry[metric] = {
                            "median": statistics.median(samples),
                            "p95": percentile(samples, 0.95),
                        }
                    breakdown.append(entry)
    summary["breakdown"] = breakdown
    paired = [
        (index[(r["case"], "quality", "off", r["seed"])], r)
        for r in quality if r["arm"] == "auto"
        and (r["case"], "quality", "off", r["seed"]) in index
    ]
    summary["diagnostic_agreement"] = {
        "pairs": len(paired),
        "status_and_confirmed_codes": sum(
            (a["status"], a["confirmed_codes"]) == (b["status"], b["confirmed_codes"])
            for a, b in paired
        ),
        "status_and_all_codes": sum(
            (a["status"], a["codes"]) == (b["status"], b["codes"]) for a, b in paired
        ),
    }
    if breakdown:
        flat = []
        for entry in breakdown:
            row = {}
            for key, value in entry.items():
                if isinstance(value, dict) and set(value) == {"median", "p95"}:
                    row.update({f"{key}_{stat}": number for stat, number in value.items()})
                else:
                    row[key] = (
                        json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
                    )
            flat.append(row)
        with (output / "breakdown.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if records:
        with (output / "measurements.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    if plots and performance:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        labels = [p["case"] for p in performance]
        for axis, key, ci_key, title in [
            (axes[0], "median_speedup", "speedup_ci95", "End-to-end speedup"),
            (axes[1], "cpu_stage_speedup", "cpu_stage_speedup_ci95", "CPU-stage speedup"),
        ]:
            axis.barh(
                labels, [p[key] for p in performance], color="#356e9a",
                xerr=[
                    [p[key] - p[ci_key][0] for p in performance],
                    [p[ci_key][1] - p[key] for p in performance],
                ], capsize=3,
            )
            axis.axvline(1, color="black", linewidth=1)
            axis.set_title(title)
            axis.set_xlabel("Baseline / automatic reduction (higher is faster)")
        axes[0].axvline(1.25, color="#a05b21", linestyle="--", label="1.25x target")
        axes[0].legend(loc="lower right")
        figure.suptitle("Five paired repetitions per workload; 95% bootstrap intervals")
        for suffix in ("png", "svg", "pdf"):
            figure.savefig(output / f"comparison.{suffix}", dpi=180)
        plt.close(figure)
        if frozen.get("corpus_kind") == "scale_study":
            figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
            by_case = {p["case"]: p for p in performance}
            for axis, family in zip(
                axes.flat, ("mlp", "cnn", "residual", "multimodal"), strict=True
            ):
                selected = [
                    c for c in frozen["cases"] if c["family"] == family and c["name"] in by_case
                ]
                widths = [c["width"] for c in selected]
                values = [by_case[c["name"]] for c in selected]
                axis.errorbar(
                    widths, [p["median_speedup"] for p in values],
                    yerr=[
                        [p["median_speedup"] - p["speedup_ci95"][0] for p in values],
                        [p["speedup_ci95"][1] - p["median_speedup"] for p in values],
                    ], marker="o", capsize=4,
                )
                axis.axhline(1, color="black", linewidth=1)
                axis.axhline(1.25, color="#a05b21", linestyle="--")
                axis.set_xscale("log", base=2)
                axis.set_xticks(widths, labels=[str(w) for w in widths])
                axis.set_title(family)
                axis.set_xlabel("Original hidden width; batch and input dimensions fixed")
                axis.set_ylabel("End-to-end speedup (off / auto)")
            figure.suptitle("Width-only follow-up; five pairs and 95% bootstrap intervals")
            for suffix in ("png", "svg", "pdf"):
                figure.savefig(output / f"scaling.{suffix}", dpi=180)
            plt.close(figure)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run", "summarize"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument("--scale-study", action="store_true", help="Freeze width-only timing cases")
    parser.add_argument(
        "--auto-only", action="store_true", help="Freeze an auto-only labeled quality audit"
    )
    parser.add_argument("--phases", default="reference,quality,timing")
    parser.add_argument(
        "--cases", help="Comma-separated cases to freeze, or a partial subset of a frozen run"
    )
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == "freeze":
        if args.auto_only and args.scale_study:
            parser.error("--auto-only and --scale-study are mutually exclusive")
        freeze(output, args.archive_root, args.scale_study, args.auto_only,
               set(args.cases.split(",")) if args.cases else None)
    elif args.action == "run":
        run(output, args.phases.split(","), set(args.cases.split(",")) if args.cases else None)
        summarize(output, False)
    else:
        summarize(output, args.plots)


if __name__ == "__main__":
    main()
