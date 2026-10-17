"""Reference two-epoch adapter for candidates exposing preflight_train_epoch.

Run this only inside a trusted CPU-only candidate environment. It is an
adapter contract, not a security sandbox or a universal training-script parser.
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import os
from pathlib import Path
import runpy
import traceback


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("script_path", type=Path)
    parser.add_argument("fixture_root", type=Path)
    parser.add_argument("epochs", type=int)
    args = parser.parse_args()
    if args.epochs != 2 or not args.fixture_root.is_dir():
        print(json.dumps({"status": "inconclusive", "reason": "two epochs and an existing fixture root are required"}))
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["NVIDIA_VISIBLE_DEVICES"] = "void"
    try:
        candidate = runpy.run_path(str(args.script_path.resolve()), run_name="__preflight__")
    except Exception as exc:
        print(json.dumps({"status": "inconclusive", "reason": "candidate_import_failed",
                          "error": str(exc)[:1000]}))
        return 0
    train_epoch = candidate.get("preflight_train_epoch")
    if not callable(train_epoch):
        print(json.dumps({"status": "inconclusive", "reason": "missing_preflight_train_epoch"}))
        return 0
    losses: list[float] = []
    for epoch in range(2):
        try:
            loss = float(train_epoch(epoch, args.fixture_root.resolve()))
            if not math.isfinite(loss):
                raise ValueError("non-finite loss")
        except Exception as exc:
            fixture_or_scratch_error = isinstance(exc, PermissionError) or (
                isinstance(exc, OSError) and exc.errno in {errno.EROFS, errno.ENOSPC}
            )
            print(json.dumps({"status": "inconclusive" if fixture_or_scratch_error else "failed",
                              "bug_confirmed": not fixture_or_scratch_error,
                              "epochs_completed": epoch,
                              "traceback": traceback.format_exc()[-4000:]}))
            return 0
        losses.append(loss)
    print(json.dumps({"status": "passed", "epochs_completed": 2, "losses": losses}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
