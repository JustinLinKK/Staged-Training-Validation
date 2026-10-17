# Model Preflight

Model Preflight decides whether an agent-generated PyTorch training candidate is worth a
GPU job. It performs conservative source checks in the controller, then imports and executes
candidate code only in fresh, resource-bounded worker processes.

A `PASS` means the configured CPU checks passed. It does **not** prove CUDA correctness,
mixed-precision stability, GPU memory feasibility, custom-kernel behavior, MPS, streams,
NCCL, DDP, or concurrent execution. Keep a short target-GPU canary at the start of the real
training job.

## Quickstart

Create and activate the dedicated environment:

```bash
conda env create -f environment.yml
conda activate model-preflight
```

For the environment already created in this checkout, install/update the editable package:

```bash
conda activate model-preflight
python -m pip install -e '.[dev]'
```

Run the included example:

```bash
model-preflight check examples/image_classification/preflight.yaml \
  --report report.json \
  --junit report.xml
```

Run an individual stage while repairing a candidate:

```bash
model-preflight check preflight.yaml --only abstract_forward --report report.json
```

The Python API returns the same report model:

```python
from model_preflight import check

report = check("preflight.yaml")
print(report.overall_status, report.gpu_submission_recommended)
```

## Checks

The default pipeline runs:

1. `static_source`: syntax, entry-module presence, device hard-coding, hard-coded batch
   dimensions, CUDA-only packages/branches, streams, distributed APIs, and custom extensions.
2. `hardware`: requested precision against a versioned target profile.
3. `construction`: adapter import, `nn.Module` construction, trainable parameters, optimizer
   type, parameter ownership, and non-empty groups.
4. `data_contract`: a representative batch from the adapter's real fixture/data hook, with
   optional input-rank and target-dtype contracts.
5. `abstract_forward`: every configured batch size, normal/boundary shape, and precision using
   `torch.export` when available plus a meta-tensor execution fallback. Export failure alone is
   never a candidate failure.
6. `cpu_training`: real scalar finite loss, backward, required/available gradients, finite
   gradients, optimizer step, optional second stateful step, normal batch, and partial final
   batch where applicable.
7. `validation`: `eval()` plus `no_grad()` through the configured validation data and step.
8. `memory`: exact parameter/gradient bytes plus explicitly labeled optimizer, saved-activation,
   and safety-margin estimates.

Every stage records scenario evidence. All runtime stages use a fresh worker, so candidate
imports and global state never enter the long-lived controller.

Experimental [hidden-width reduction](docs/shape-reduction.md) can run CPU workflow checks on
smaller models after original-shape checks pass. It is disabled by default and reports reduced
execution explicitly; full-width numerical behavior still requires validation.
Supported reductions include PyTorch Transformer feed-forward widths and Hugging Face Mamba-1
CPU expansions. Unsupported layers retain the entire original model; see the
[architecture coverage and limits](docs/shape-reduction.md#transformer-and-mamba-coverage).
The [adversarial audit](docs/shape-reduction-audit.md) tracks reduction blind spots and
regression coverage using the automatic arm only.

## Outcomes and exit codes

| Outcome | Exit | Meaning |
|---|---:|---|
| `PASS` | 0 | Every selected required CPU check passed. |
| `FAIL` | 10 | A candidate defect was reproduced. |
| `INCONCLUSIVE` | 20 | A property could not be checked, such as a timeout or missing meta kernel. |
| invalid manifest/profile | 30 | Input contract failed before candidate execution. |
| `INTERNAL_ERROR` | 40 | The checker/worker protocol failed; this is never a candidate defect. |

Policy modes change only the scheduling recommendation:

- `audit` never recommends blocking a submission.
- `balanced` blocks confirmed failures and allows inconclusive results with
  `GPU_CHECK_REQUIRED`.
- `strict` recommends submission only for `PASS`.

## Candidate adapter

The manifest names an explicit `module:attribute`; Model Preflight does not guess training entry
points. The object must implement:

```python
class CandidateAdapter:
    def build_model(self, context): ...
    def build_optimizer(self, model, context): ...
    def build_train_batch(self, scenario, device): ...
    def build_validation_batch(self, scenario, device): ...
    def training_step(self, model, batch, context): ...  # scalar loss tensor
    def validation_step(self, model, batch, context): ...
```

The batch builders must use the real transform/tokenizer/dataset/collate path wherever one
exists. They receive `"cpu"` for real checks and `"meta"` for abstract checks. An optional
`required_gradient_names(model)` method can identify parameters that must receive a gradient.
See [docs/adapter-api.md](docs/adapter-api.md).

## Manifest

```yaml
schema_version: 1
candidate:
  id: candidate_0182
  root: ./candidate
  adapter: preflight_adapter:CandidateAdapter
task:
  name: image_classification
  fixture_root: ./fixtures
  num_classes: 2
  target_dtype: int64
  input_rank: 3
  output_class_dimension: -1
scenarios:
  train_batch_sizes: [8, 16, 32, 64, 128]
  test_last_batch: true
  last_batch_size: 1
  run_validation: true
  input_shapes:
    normal: [3, 96, 96]
  boundary_shapes:
    wide: [3, 96, 128]
  precision: [fp32, fp16]
target:
  profile: nvidia/v100_32gb
execution:
  abstract_timeout_seconds: 30
  cpu_timeout_seconds: 90
  maximum_cpu_memory_mb: 8192
  maximum_processes: 32
  maximum_output_bytes: 1000000
  disable_network: true
  allow_real_cpu_abstract_fallback: false
  cache: false
policy:
  mode: balanced
```

Relative paths resolve against the manifest directory. Unknown versions and fields are rejected
before execution. See [docs/manifest.md](docs/manifest.md) and the packaged JSON Schema.

## Isolation

Local workers receive a clean temporary home, no visible CUDA device, a redacted environment,
offline library flags, a Python network audit hook, process-group cleanup, and POSIX wall-time,
CPU, address-space, process-count, core-dump, and file-size limits. Candidate stdout/stderr are
captured away from the protocol.

OS subprocess limits are defense in depth, not a complete sandbox against hostile native code.
For untrusted generated code, use the CPU container with a read-only candidate mount, read-only
root filesystem, no capabilities, a PID limit, and no network:

```bash
docker build -f docker/Dockerfile.cpu -t model-preflight:cpu .
docker run --rm --network none --read-only --cap-drop ALL --pids-limit 32 \
  --tmpfs /tmp:rw,noexec,nosuid,size=1g \
  -v "$PWD/examples/image_classification:/work:ro" \
  -v "$PWD/reports:/reports:rw" \
  model-preflight:cpu check /work/preflight.yaml --report /reports/report.json
```

## Development

```bash
conda activate model-preflight
ruff check .
mypy src/model_preflight
python -m pytest
python -m build
```

The test suite includes schema/unit tests, subprocess timeout/crash isolation tests, CLI tests,
and a mutation corpus with pre-recorded expected outcomes. Historical-corpus metrics must be
measured with real MLEvolve artifacts using `scripts/evaluate_bug_corpus.py`; no historical
artifacts are bundled or invented here.

Architecture, schemas, diagnostics, profiles, and integration guidance live under
[docs/](docs/architecture.md). The honest phase-by-phase release-gate status is recorded in
[docs/implementation-status.md](docs/implementation-status.md).
