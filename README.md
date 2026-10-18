# Staged Training Validation

CPU-first preflight checks for generated PyTorch training candidates. Give the checker a
candidate manifest and a small representative fixture; receive a structured diagnostic report
before spending a full GPU allocation.

**Paper software release: v0.2.1.** Install from this repository or a built wheel. This project
is not published to PyPI; the public `model-preflight` PyPI project is unrelated.

[Documentation wiki](docs/README.md) · [Installation](docs/installation.md) ·
[Agent integration](docs/agent-integration.md) · [Paper release and evidence](docs/paper-release.md) ·
[Changelog](CHANGELOG.md)

## Install

Use Python 3.10+ on Linux or WSL2, preferably in a virtual environment. From a clone:

```bash
git clone https://github.com/JustinLinKK/nn-model-preflight-checker.git
cd nn-model-preflight-checker
git checkout v0.2.1
python -m pip install .
```

Or install the paper release directly, without managing a checkout:

```bash
python -m pip install "git+https://github.com/JustinLinKK/nn-model-preflight-checker.git@v0.2.1"
```

A compatible PyTorch installation is reused. For a new CPU-only environment, install PyTorch
from its CPU index before installing this package; see [installation options](docs/installation.md).
Candidate-specific dependencies are supplied by the calling environment.

## Run a check

From the repository, run the included synthetic example:

```bash
model-preflight check examples/image_classification/preflight.yaml --report report.json
```

From any installed environment, use your own manifest:

```python
from model_preflight import check

report = check("preflight.yaml")
print(report.overall_status.value)
print(report.gpu_submission_recommended)
```

The manifest names a `module:CandidateAdapter`; the adapter builds the model, optimizer,
training/validation batches, and real loss. See the [adapter contract](docs/adapter-api.md) and
[manifest reference](docs/manifest.md). Runtime imports and execution stay in fresh worker
processes. The controller never imports candidate code.

## Speculative integration

```python
from model_preflight import submit

submission = submit("preflight.yaml")
print(submission.initial.status)  # STATIC_PASS, or an immediate terminal rejection
# The framework can queue the candidate and continue working here.
current = submission.poll()      # nonblocking
final = submission.completion.result()  # wait only when the final decision is needed
```

For CPU container workers, stream the same lifecycle as JSON lines:

```bash
model-preflight check preflight.yaml --format events
```

`STATIC_PASS` is provisional. The complete workflow later returns `FULL_PASS`, `FAIL`,
`INCONCLUSIVE`, or `INTERNAL_ERROR`. The framework correlates request IDs with scheduler job
IDs and handles late rejection. [Integration details](docs/agent-integration.md) cover Docker,
immutable inputs, lost workers, and queue policy. Local package use avoids launching a Docker
container per request; PyTorch worker startup and simulation costs remain.

## Checking pipeline and options

The full pipeline runs source and hardware checks, original-width construction and data
contracts, all original-shape abstract scenarios, CPU training/backward/optimizer checks,
validation, and an original-model memory estimate. See [architecture](docs/architecture.md).

```yaml
execution:
  max_workers: 4
  cpu_threads: 1
  shape_reduction:
    mode: off
```

Concurrency is optional; both worker and thread counts default to one. Experimental
[shape reduction](docs/shape-reduction.md) can shrink supported hidden widths after original-shape
checks pass. It includes eligible Transformer feed-forward and Mamba-1 CPU paths. Unsupported
structures retain original width. Reduction is **disabled by default**, and reduced success
still requires original-width numerical validation.

| Report outcome | CLI exit | Meaning |
|---|---:|---|
| `PASS` | 0 | Configured required CPU checks passed. |
| `FAIL` | 10 | A candidate defect was reproduced. |
| `INCONCLUSIVE` | 20 | Verification could not complete or establish a property. |
| Invalid manifest/profile | 30 | Input contract failed before candidate execution. |
| `INTERNAL_ERROR` | 40 | Checker infrastructure failed; not candidate blame. |

Policy modes change scheduling recommendations: `audit` always recommends admission,
`balanced` admits inconclusive results with a GPU-check requirement, and `strict` admits only
`PASS`. A CPU pass does not establish CUDA correctness, GPU memory feasibility, mixed-precision
stability, or distributed-training correctness. Keep a target-GPU canary.

## Docker and trust boundary

Use the Python package directly for trusted local candidates. For untrusted generated code,
keep the existing CPU container and runtime isolation controls:

```bash
docker build -f docker/Dockerfile.cpu -t model-preflight:cpu .
docker run --rm --init --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 128 --cpus 4 --memory 4g \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g \
  --mount type=bind,src="$PWD/examples/image_classification",dst=/work,readonly \
  model-preflight:cpu check /work/preflight.yaml --format events
```

Subprocess limits alone are not a sandbox against hostile native code. See the
[security model](docs/security.md). The separate root `Dockerfile` and root compatibility
modules serve the [Lightning integration](README-lightning.md); they retain their existing
paths and are not the `model_preflight` package API.

## Research and development

The [paper-release guide](docs/paper-release.md) separates measured results from limitations
and links reproduction commands. Historical experiment results are not measurements of every
later revision. In particular, the original shape-reduction acceptance gates failed; later
auto-only audits do not establish universal numerical equivalence.

Use [CITATION.cff](CITATION.cff) to cite the software, pin `v0.2.1` or its commit, and record the
PyTorch version and execution settings. See [development](docs/development.md) for editable
installation, tests, package validation, and the repository layout.
