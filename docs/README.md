# Documentation wiki

This wiki is versioned with the code. For the paper, read it from tag **v0.2.1** rather than
the moving `main` branch. The [project README](../README.md) is the short entry point.

## Start here

| Goal | Guide |
|---|---|
| Install from a clone, Git tag, or wheel | [Installation](installation.md) |
| Connect an agent or GPU queue | [Agent integration and speculative admission](agent-integration.md) |
| Wrap a model and real task fixture | [Candidate adapter](adapter-api.md) |
| Configure scenarios, workers, and threads | [Manifest](manifest.md) |
| Understand the checking stages | [Architecture](architecture.md) |
| Interpret failures and recommendations | [Diagnostics](diagnostics.md) |
| Run generated code safely | [Security and Docker](security.md) |

## Features and reference

- [Shape reduction](shape-reduction.md): eligibility, Transformer/Mamba coverage, full-width
  fallback, and numerical limits.
- [GPU profiles](gpu-profiles.md): target capabilities and analytical memory estimates.
- [Implementation status](implementation-status.md): implemented capabilities and open gates.
- [Lightning compatibility](../README-lightning.md): the independent source/probe interface.
- [Development](development.md): repository map, tests, builds, and installed-package checks.

## Paper and experiments

Start with the [paper-release guide](paper-release.md), which states what can and cannot be
claimed from the available evidence.

| Study | Report |
|---|---|
| Original paired reduction comparison | [Design, corpus, and failed acceptance gates](shape-reduction-experiment.md) |
| Larger models and the overflow fix | [Width-scaling follow-up](shape-reduction-followup.md) |
| Shrink-induced training defects | [Auto-only adversarial audit](shape-reduction-audit.md) |
| Transformer/Mamba and unsupported controls | [Architecture coverage](shape-reduction.md#transformer-and-mamba-coverage) |
| Multicore timing and diagnostic parity | [Parallel execution measurements](manifest.md#parallel-execution) |
| Provisional response and late rejection | [Integration verification](agent-integration.md#verification) |

The [initial design plan](history/initial-plan.md) is retained as history, not a current feature
promise. [CHANGELOG.md](../CHANGELOG.md) records release changes;
[CITATION.cff](../CITATION.cff) identifies the software citation.
