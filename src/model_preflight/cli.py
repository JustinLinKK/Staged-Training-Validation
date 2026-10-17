"""Command-line interface and stable exit codes."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from model_preflight.api import check
from model_preflight.core.enums import OverallStatus
from model_preflight.core.exceptions import ManifestError
from model_preflight.engine.registry import STAGE_ORDER
from model_preflight.reporting.json_report import render_json, write_json
from model_preflight.reporting.junit_report import write_junit
from model_preflight.reporting.text_report import render_text
from model_preflight.speculative import submit
from model_preflight.version import __version__

EXIT_CODES = {
    OverallStatus.PASS: 0,
    OverallStatus.FAIL: 10,
    OverallStatus.INCONCLUSIVE: 20,
    OverallStatus.INTERNAL_ERROR: 40,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="model-preflight",
        description="CPU-first validation for PyTorch training candidates",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("check", help="check one preflight manifest")
    command.add_argument("manifest")
    command.add_argument("--report", help="write the versioned JSON report")
    command.add_argument("--junit", help="write a JUnit XML report")
    command.add_argument(
        "--format",
        choices=("text", "json", "events"),
        default="text",
        help="stdout format; events streams provisional admission then the final result as JSONL",
    )
    command.add_argument(
        "--only",
        action="append",
        choices=STAGE_ORDER,
        help="run only this stage; may be repeated",
    )
    command.add_argument("--no-cache", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.format == "events" and arguments.only:
        parser.error("--format events requires the complete workflow; --only is not supported")
    try:
        if arguments.format == "events":
            submission = submit(arguments.manifest)
            print(json.dumps(submission.initial.to_dict()), flush=True)
            final = submission.completion.result()
            if not submission.initial.terminal:
                print(json.dumps(final.to_dict()), flush=True)
            if final.report is None:
                return EXIT_CODES[OverallStatus(final.status)]
            report = final.report
        else:
            report = check(
                arguments.manifest,
                only=set(arguments.only) if arguments.only else None,
                use_cache=False if arguments.no_cache else None,
            )
    except ManifestError as exc:
        print(f"model-preflight: invalid manifest: {exc}", file=sys.stderr)
        return 30
    except Exception as exc:
        print(
            f"model-preflight: controller internal error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 40
    if arguments.report:
        write_json(report, arguments.report)
    if arguments.junit:
        write_junit(report, arguments.junit)
    if arguments.format != "events":
        print(render_json(report) if arguments.format == "json" else render_text(report), end="")
    return EXIT_CODES[report.overall_status]


if __name__ == "__main__":
    raise SystemExit(main())
