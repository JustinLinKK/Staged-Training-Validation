# Changelog

## 0.2.0 - 2026-10-17

Research software release for the paper, installed from this repository or its wheel.
No public PyPI publication is part of this release.

- Add opt-in hidden-width reduction after original-shape checks, conservative full-width
  fallback, and original-width confirmation of failed reduced workflows.
- Support eligible PyTorch Transformer feed-forward widths and Hugging Face Mamba-1 CPU
  expansions; preserve unsupported architectures at original width.
- Check nested validation results and post-update parameters/optimizer state; guard against
  the width-dependent defects reproduced by the adversarial audit.
- Add bounded scenario concurrency and configurable CPU threads with isolated workers.
- Add nonblocking `submit()` and JSONL speculative admission events, preserving the blocking
  `check()` API, existing CLI, manifest/report schemas, and Lightning compatibility interface.
- Prepare repository/wheel installation, a versioned documentation wiki, software citation,
  and research evidence/reproduction guidance. Keep the existing Docker images and defaults.

Reduction remains disabled by default. The original reduction comparison failed its quality
and end-to-end performance gates; later audits and concurrency measurements have narrower
claims. See `docs/paper-release.md`.

## 0.1.0 - Development baseline

- Add the standalone PyTorch preflight controller, isolated worker protocol, schemas, CLI,
  static checks, runtime checks, hardware profiles, and transparent memory-risk estimate.
- Require an isolated, CPU-only candidate probe image before speculative scheduler admission;
  use read-only source mounts and writable scratch, remove the exact probe container on timeout,
  and treat read-only fixture failures as inconclusive rather than candidate defects.

