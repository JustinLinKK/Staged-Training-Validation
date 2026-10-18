# Paper software release: v0.2.0

Release date: **2026-10-17**. This is a research software snapshot, not a claim of universal
model correctness or completed GPU validation. Cite [CITATION.cff](../CITATION.cff), pin the
release tag or full commit, and record the actual environment used for each experiment.

Install from the repository with `python -m pip install .`, or use the
[tagged Git installation](installation.md#install-directly-from-git). There is no public PyPI
publication. Existing Dockerfiles and the `model_preflight` import/`model-preflight` CLI remain
available. Manifest/report schema and worker protocol identifiers remain at version 1.

## Released capabilities

- Original-shape source, hardware, construction, data-contract, and abstract scenario checks.
- CPU training/loss/backward/optimizer and separate validation checks with isolated workers.
- Opt-in hidden-width reduction, conservative fallback, and original-width confirmation of
  failed reduced execution. Supported Transformer/Mamba paths are explicitly bounded.
- Configurable scenario concurrency and per-worker CPU threads; defaults remain serial.
- Provisional `STATIC_PASS`, background workflow checking, and a correlated terminal update.
  Only `FULL_PASS` establishes completion of the configured CPU workflow.

Reduction defaults to **off**. Python installation avoids per-request Docker launch overhead
for trusted local use; it retains subprocess startup and simulation work. It does not replace
the external sandbox required for untrusted candidate code.

## Release verification

On 2026-10-17, the full suite passed **233 tests**, with six explicit skips for unavailable
native `mamba_ssm` and optional compatibility probe images. Ruff, mypy, and source/wheel builds
passed. In a fresh Python 3.13 environment with PyTorch 2.14.1+cpu, `pip install .` passed
dependency checks and the documented example passed all eight stages outside the checkout.
The installed wheel also produced `STATIC_PASS` followed by `FULL_PASS` for a reduced MLP
with two workers/two training steps, and a late `FAIL` (`AUT001`) for detached loss. Both
reports passed schema validation. These are installation and behavior checks, not new timing
measurements or target-GPU validation.

## What the evidence establishes

| Study | Observed result | Interpretation |
|---|---|---|
| [Original reduction comparison](shape-reduction-experiment.md), revision 2 | 416 reports; defect recall 69/69 off versus 66/69 auto; 0.934x median end-to-end ratio | Both original quality and speed gates failed. |
| [Overflow fix and width scaling](shape-reduction-followup.md), revision 3 | The three missed overflow runs were detected through full-width fallback; larger convolutional workloads benefited | Selected defect repair and size effects; not a rerun of all original gates. |
| [Adversarial audit](shape-reduction-audit.md), revision 4 | 180 runs matched frozen labels; 123/123 labeled defects detected | Auto-only synthetic coverage, not baseline-relative or universal equivalence. |
| [Architecture replay](shape-reduction.md#transformer-and-mamba-coverage), revision 5 | 72 runs across 24 cases: 54 PASS, 15 FAIL, 3 INCONCLUSIVE | Supported reductions and unsupported-path fallback; unavailable native Mamba excluded. |
| [Concurrency comparison](manifest.md#parallel-execution) | 27 valid runs; four workers gave 1.78x median paired speedup; nine defect runs retained outcomes | Three synthetic workloads on one shared host, with v2 enabled in every arm. |
| [Speculative integration](agent-integration.md#verification) | Eight local/container cases matched their expected outcomes; container MLP returned at 0.439 s and completed at 8.105 s | Demonstrates response/completion overlap; scheduler queue was simulated. |

These studies measured different frozen implementations. Their recorded source hashes and
reducer revisions identify the relevant code; do not relabel an older study as a measurement
of every later release. `FULL_PASS` does not establish target-GPU or full-width numerical
correctness when reduced execution was used. Inconclusive outcomes are not defect detections.

The original archived candidates used uncertain labels and, where available, synthetic
fixtures. They are separate from synthetic diagnostic-recall results and are not evidence of
real-data validation. No reduction study proves behavior over long training histories.

## Reproduce on the release

Use the development environment described in [development.md](development.md). Run from the
repository root and use new output directories. These commands evaluate the checked-out
release, not the older frozen implementations quoted above.

```bash
# Auto-only synthetic quality replay; original-width fallback remains part of auto mode.
PYTHONPATH=src:. python scripts/run_shape_reduction_experiment.py freeze \
  --auto-only --output .cache/paper-v020-auto
PYTHONPATH=src:. python scripts/run_shape_reduction_experiment.py run \
  --output .cache/paper-v020-auto
PYTHONPATH=src:. python scripts/run_shape_reduction_experiment.py summarize \
  --output .cache/paper-v020-auto

# Compare concurrency configurations, all using v2 automatic reduction.
PYTHONPATH=src:. python scripts/benchmark_parallel_execution.py \
  --output .cache/paper-v020-parallel

# Exercise the installed package and provisional/final event interface.
model-preflight check examples/image_classification/preflight.yaml --format events
```

The original paired reduction experiment is documented separately; no new reduction-off
baseline is required by this release packaging work. For plots, install Matplotlib in the
analysis environment. Preserve sources, labels, manifests, seeds, package versions, host load,
timeouts, cache/thread settings, raw reports, and excluded cases with the results.

## Artifact availability and limits

This Git release and its source distribution contain the checker, experiment generators,
tests, synthetic example, documentation, citation metadata, and published figures. The runtime
wheel contains only the installable package and resources needed by the checker.

Historical raw reports and frozen workspaces referenced by `.cache/` links in the study
reports remain local artifacts; they are not bundled into the Git repository or wheel.
Those links resolve only in a workspace with the retained artifacts. Figures are included
under `docs/figures/` so the documentation renders without that workspace. Exact replay of an
older study requires its frozen implementation and inputs, not merely the v0.2.0 tag.

The release is intentionally separate from any paper bibliographic record: no paper title,
author list, DOI, acceptance, or archival DOI is invented here. Replace the software citation
with an appropriate combined citation when the paper and archival record are available.
