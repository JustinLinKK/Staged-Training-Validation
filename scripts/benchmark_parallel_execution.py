#!/usr/bin/env python3
"""Compare serial and parallel v2 checks on identical synthetic candidates."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from run_shape_reduction_experiment import CONTROLLER, REPO, code_hash
from shape_reduction_corpus import cases, manifest

ARMS = {"serial": (1, 1), "workers4": (4, 1), "workers2_threads2": (2, 2)}


def signature(report: dict[str, Any]) -> Any:
    return (
        report["overall_status"], report.get("gpu_submission_recommended"),
        report.get("gpu_check_required"), report.get("resource_estimate"),
        [(s["name"], s["status"]) for s in report.get("stages", [])],
        [(d["stage"], d["code"], d["classification"], d.get("scenario"))
         for d in report.get("diagnostics", [])],
        [e.get("shape_reduction", {}).get("applied", False)
         for s in report.get("stages", [])
         for e in s["evidence"].get("scenarios", [s["evidence"]])],
    )


def main() -> None:
    corpus = {case["name"]: case for case in cases()}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cases", nargs="+", choices=sorted(corpus), default=[
        "mlp_2048", "cnn_128", "architecture_transformer_encoder",
    ])
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(REPO / "src", output / "implementation" / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copytree(REPO / "scripts", output / "implementation" / "scripts",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    frozen = {
        "implementation_sha256": code_hash(), "python": sys.version,
        "affinity": sorted(os.sched_getaffinity(0)), "arms": ARMS,
        "cases": args.cases, "repetitions": args.repetitions,
        "maximum_processes": 4096, "shape_reduction": "auto", "cache": False,
        "platform": platform.platform(), "torch": importlib.metadata.version("torch"),
        "cpu_model": next((line.split(":", 1)[1].strip()
                           for line in Path("/proc/cpuinfo").read_text().splitlines()
                           if line.startswith("model name")), "unknown"),
    }
    (output / "experiment.json").write_text(json.dumps(frozen, indent=2) + "\n")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(output / "implementation" / "src")
    rows = []
    signatures: dict[tuple[str, int, str], Any] = {}
    for name in args.cases:
        case = corpus[name]
        candidate = output / "corpus" / name
        candidate.mkdir(parents=True)
        (candidate / "candidate.py").write_text(case["source"])
        for repetition in range(args.repetitions):
            arms = list(ARMS)
            arms = arms[repetition % len(arms):] + arms[:repetition % len(arms)]
            for arm in arms:
                trial = output / "raw" / name / f"{repetition}-{arm}"
                trial.mkdir(parents=True)
                configured = manifest(case, candidate)
                batch = configured["scenarios"]["train_batch_sizes"][0]
                configured["scenarios"]["train_batch_sizes"] = [batch * i for i in (1, 2, 3, 4)]
                configured["execution"].update(
                    max_workers=ARMS[arm][0], cpu_threads=ARMS[arm][1],
                    maximum_processes=4096, seed=[11, 23, 37][repetition % 3],
                    shape_reduction={"mode": "auto"}, cache=False,
                )
                config_path, report_path = trial / "manifest.json", trial / "report.json"
                config_path.write_text(json.dumps(configured, indent=2))
                before_load = os.getloadavg()[0]
                started = time.monotonic()
                with (trial / "controller.log").open("w") as log:
                    subprocess.run(
                        [sys.executable, "-c", CONTROLLER, str(config_path), str(report_path)],
                        env=environment, stdout=log, stderr=log, check=True, timeout=900,
                    )
                elapsed = time.monotonic() - started
                report = json.loads(report_path.read_text())
                signatures[name, repetition, arm] = signature(report)
                evidence = [e for s in report.get("stages", [])
                            for e in s["evidence"].get("scenarios", [s["evidence"]])]
                stages = {s["name"]: s for s in report.get("stages", [])}
                codes = {d["code"] for d in report.get("diagnostics", [])}
                row = {
                    "case": name, "repetition": repetition, "arm": arm,
                    "seconds": elapsed, "status": report["overall_status"],
                    "expected_label": report["overall_status"] == case["expected_status"]
                    and set(case.get("expected_codes", [])) <= codes,
                    "abstract_seconds": stages.get("abstract_forward", {}).get("duration_seconds"),
                    "cpu_seconds": stages.get("cpu_training", {}).get("duration_seconds"),
                    "training_losses": [s.get("loss") for s in stages.get(
                        "cpu_training", {},
                    ).get("evidence", {}).get("scenarios", [])],
                    "applied_scenarios": sum(e.get("shape_reduction", {}).get("applied", False)
                                             for e in evidence),
                    "max_worker_rss_bytes": max((e.get("peak_worker_rss_bytes", 0)
                                                 for e in evidence), default=0),
                    "load_before": before_load, "load_after": os.getloadavg()[0],
                }
                rows.append(row)
                (trial / "measurement.json").write_text(json.dumps(row, indent=2) + "\n")
                print(json.dumps(row), flush=True)
    with (output / "measurements.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    index = {(r["case"], r["repetition"], r["arm"]): r for r in rows}
    summary = {
        "runs": len(rows), "labels_match": all(r["expected_label"] for r in rows),
        "diagnostic_and_reduction_parity": all(
            signatures[name, rep, arm] == signatures[name, rep, "serial"]
            for name, rep, arm in signatures
        ),
        "fixed_thread_loss_parity": all(
            index[name, rep, "serial"]["training_losses"]
            == index[name, rep, "workers4"]["training_losses"]
            for name in args.cases for rep in range(args.repetitions)
        ),
        "cases": {
            name: {
                arm: {
                    "median_seconds": statistics.median(
                        index[name, rep, arm]["seconds"] for rep in range(args.repetitions)
                    ),
                    "median_paired_speedup": statistics.median(
                        index[name, rep, "serial"]["seconds"] / index[name, rep, arm]["seconds"]
                        for rep in range(args.repetitions)
                    ),
                } for arm in ARMS
            } for name in args.cases
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    if not all(summary[key] for key in (
        "labels_match", "diagnostic_and_reduction_parity", "fixed_thread_loss_parity",
    )):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
