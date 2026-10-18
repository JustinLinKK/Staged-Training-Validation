# Model Preflight: Standalone Repository Implementation Plan

Historical design document. For the released interface and measured limitations, start at
the [documentation wiki](../README.md) and [paper release](../paper-release.md).

## 1. Mission

Build a new standalone repository, working name `model-preflight`, that determines whether an agent-generated PyTorch training candidate is sufficiently valid to justify submitting a GPU job.

The checker must catch as many source, model-structure, data-contract, loss, backward, optimizer, validation, batch-size, and target-hardware mistakes as possible on CPU before MLEvolve creates a GPU job. The repository will later be added to MLEvolve as a pinned Git submodule and must remain portable to other agentic frameworks.

The project is a preflight checker, not a GPU emulator. It must never claim that CPU validation proves CUDA correctness, true GPU memory feasibility, numerical stability under mixed precision, or correct MPS/stream/concurrent execution.

## 2. Required Outcome

At completion, a caller must be able to run:

```bash
model-preflight check preflight.yaml --report report.json
```

The command must:

1. Validate the manifest and candidate source without executing candidate code.
2. Import and construct the candidate in an isolated worker.
3. Validate a tiny representative batch produced through the real task data path.
4. Test forward, output, and loss contracts using abstract tensors at production shapes.
5. Run a real CPU micro-training step: forward, scalar loss, backward, gradient checks, and optimizer step.
6. Run the validation/inference path when configured.
7. Test every relevant batch-size and boundary-shape scenario.
8. Compare candidate requirements with a versioned target-GPU profile.
9. Estimate resource risk without presenting the estimate as a guarantee.
10. Produce a stable, machine-readable report suitable for review-agent repair and scheduler admission.

The final status must be one of:

```text
PASS
FAIL
INCONCLUSIVE
INTERNAL_ERROR
```

Meanings:

- `PASS`: all required CPU checks passed; GPU-specific risk may still remain.
- `FAIL`: a confirmed candidate defect was reproduced.
- `INCONCLUSIVE`: the checker could not validate a property, such as a CUDA-only operator or a timed-out CPU check.
- `INTERNAL_ERROR`: the checker itself failed. This must never be reported as a candidate defect.

## 3. Hard Requirements

### 3.1 Repository independence

- Do not import MLEvolve, PerfSeer, LangGraph, Codex, AutoGen, or another agent framework in the checker core.
- Expose a CLI, Python API, versioned manifest schema, and versioned JSON report schema.
- Keep MLEvolve translation and repair-loop code in the parent MLEvolve repository.
- Design framework backends through protocols so JAX or TensorFlow could be added later, but implement PyTorch only in version 1.

### 3.2 Dependency policy

- Do not integrate TorchShapeFlow, PyTea, or another rarely used tensor-analysis library.
- It is acceptable and expected to use Python's standard library and common PyTorch facilities such as `torch.export`, meta tensors, FakeTensor support, autograd anomaly detection, and module hooks.
- Do not manually reimplement shape semantics for every PyTorch operator. Prefer PyTorch's own abstract/meta semantics.
- Wrap unstable or version-sensitive PyTorch functionality behind internal compatibility interfaces.

### 3.3 Correct failure classification

The following distinctions are mandatory:

```text
Confirmed matmul mismatch                 -> FAIL
Confirmed target/label dtype mismatch     -> FAIL
Missing fake kernel for a custom op       -> INCONCLUSIVE
CPU micro-step timeout                    -> INCONCLUSIVE
CUDA-only behavior                        -> INCONCLUSIVE / GPU_CHECK_REQUIRED
Checker worker crash                      -> INTERNAL_ERROR
```

- Never classify a timeout alone as a buggy model.
- Never classify an unsupported checker feature as a buggy model.
- Never let an export or tracing failure automatically become a candidate failure.
- Preserve the original exception and stack trace as evidence, but normalize the top-level diagnostic.

### 3.4 Security and isolation

Agent-generated Python is untrusted code.

- Never import candidate code into the long-lived controller process.
- Execute runtime stages in fresh subprocesses.
- Provide a CPU container that runs as non-root, without a GPU and without network access.
- Use read-only candidate mounts and a dedicated writable temporary directory.
- Enforce wall-time, memory, CPU, process-count, and output-size limits.
- Capture stdout, stderr, exit status, signals, timeout cause, and resource usage.
- Ensure a candidate cannot patch the controller's PyTorch process or leak global state into the next candidate.

## 4. Explicit Non-Goals for Version 1

- Do not simulate CUDA kernel execution.
- Do not predict training performance or colocation slowdown.
- Do not replace the target-GPU canary.
- Do not prove that CUDA OOM cannot happen.
- Do not test CUDA MPS, streams, NCCL, DDP, or concurrent processes on CPU.
- Do not auto-edit candidate files inside the checker.
- Do not require complete training datasets.
- Do not implement every ML framework in the first release.
- Do not require Kubernetes in the standalone repository.
- Do not make successful `torch.export` a requirement for a valid candidate.

## 5. High-Level Architecture

```text
Candidate source + task fixture + preflight manifest
                         |
                         v
                Preflight controller
                         |
        +----------------+----------------+
        |                |                |
        v                v                v
  Static checks    Abstract checks   CPU runtime checks
        |                |                |
        +----------------+----------------+
                         |
                         v
              Target profile checks
                         |
                         v
            Policy and report generator
                         |
                         v
       PASS / FAIL / INCONCLUSIVE / INTERNAL_ERROR
```

The controller is responsible only for orchestration, policy, caching, and reporting. Candidate imports and execution occur in short-lived workers communicating through a JSON-compatible protocol.

## 6. Proposed Repository Layout

Create the repository with the following intended structure. Minor naming changes are acceptable if the architectural boundaries remain intact.

```text
model-preflight/
├── pyproject.toml
├── README.md
├── LICENSE
├── CHANGELOG.md
├── src/
│   └── model_preflight/
│       ├── __init__.py
│       ├── api.py
│       ├── cli.py
│       ├── version.py
│       ├── core/
│       │   ├── enums.py
│       │   ├── diagnostics.py
│       │   ├── results.py
│       │   ├── policies.py
│       │   └── exceptions.py
│       ├── manifest/
│       │   ├── models.py
│       │   ├── loader.py
│       │   ├── validator.py
│       │   └── schema_v1.json
│       ├── engine/
│       │   ├── runner.py
│       │   ├── stage.py
│       │   ├── registry.py
│       │   ├── cache.py
│       │   └── context.py
│       ├── checks/
│       │   ├── source/
│       │   ├── runtime/
│       │   ├── abstract/
│       │   ├── hardware/
│       │   └── numeric/
│       ├── execution/
│       │   ├── subprocess_runner.py
│       │   ├── resource_limits.py
│       │   ├── timeout.py
│       │   ├── worker_protocol.py
│       │   └── worker_main.py
│       ├── adapters/
│       │   ├── protocol.py
│       │   ├── pytorch.py
│       │   └── function_adapter.py
│       ├── profiles/
│       │   ├── schema.json
│       │   └── nvidia/
│       │       ├── v100_16gb.yaml
│       │       ├── v100_32gb.yaml
│       │       ├── a10_24gb.yaml
│       │       └── a100_40gb.yaml
│       └── reporting/
│           ├── json_report.py
│           ├── text_report.py
│           ├── junit_report.py
│           └── stacktrace.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── golden/
│   ├── mutations/
│   └── historical/
├── examples/
│   ├── image_classification/
│   ├── transformer/
│   ├── segmentation/
│   ├── gnn/
│   └── custom_operator/
├── docker/
│   ├── Dockerfile.cpu
│   └── Dockerfile.cuda-build
├── docs/
│   ├── architecture.md
│   ├── manifest.md
│   ├── adapter-api.md
│   ├── diagnostics.md
│   ├── gpu-profiles.md
│   └── agent-integration.md
└── scripts/
    ├── evaluate_bug_corpus.py
    └── run_compatibility_matrix.py
```

## 7. Public Candidate Adapter

Define a small protocol that makes the actual training path explicit. Do not infer arbitrary training entry points when an adapter is available.

```python
from typing import Any, Protocol


class CandidateAdapter(Protocol):
    def build_model(self, context: Any) -> Any:
        ...

    def build_optimizer(self, model: Any, context: Any) -> Any:
        ...

    def build_train_batch(self, scenario: Any, device: str) -> Any:
        ...

    def build_validation_batch(self, scenario: Any, device: str) -> Any:
        ...

    def training_step(self, model: Any, batch: Any, context: Any) -> Any:
        """Return a scalar loss tensor."""
        ...

    def validation_step(self, model: Any, batch: Any, context: Any) -> Any:
        ...
```

Also provide a function-based adapter for simple candidates. The adapter must be part of the candidate package or task integration, not hardcoded into the checker.

## 8. Manifest Schema

Implement and version a declarative manifest. A representative manifest is:

```yaml
schema_version: 1

candidate:
  id: candidate_0182
  root: ./candidate
  adapter: preflight_adapter:CandidateAdapter

task:
  name: histopathologic_cancer
  num_classes: 2
  fixture_root: ./fixtures/histopathologic_cancer

scenarios:
  train_batch_sizes: [8, 16, 32, 64, 128]
  test_last_batch: true
  run_validation: true
  input_shapes:
    normal: [3, 96, 96]
  precision: [fp32, fp16]

target:
  profile: nvidia/v100_32gb

execution:
  abstract_timeout_seconds: 30
  cpu_timeout_seconds: 90
  maximum_cpu_memory_mb: 8192
  disable_network: true

policy:
  mode: balanced
```

Requirements:

- Validate manifests before candidate execution.
- Reject unknown schema versions with a manifest error, not a candidate error.
- Resolve relative paths against the manifest directory.
- Do not silently fill ambiguous task or candidate entry points.
- Store the normalized manifest in the final report.

## 9. Diagnostic and Report Schema

Diagnostics require stable codes. Begin with this taxonomy:

```text
SRC001  Python syntax error
SRC002  Import cannot be resolved
DEV001  Hard-coded CUDA device
DEV002  Direct .cuda() call
BAT001  Hard-coded batch dimension
DAT001  Dataset/model input mismatch
DAT002  Target dtype mismatch
SHP001  Tensor rank mismatch
SHP002  Matrix multiplication mismatch
SHP003  Residual shape mismatch
SHP004  Invalid reshape
OUT001  Output class mismatch
LOS001  Loss input/target mismatch
AUT001  Loss has no gradient
AUT002  Backward failure
AUT003  Missing required gradient
OPT001  Optimizer contains no parameters
NUM001  Non-finite loss
NUM002  Non-finite gradient
GPU001  Unsupported target dtype
GPU002  Unsupported compute capability
GPU003  CUDA-only behavior requires GPU validation
MEM001  Estimated memory exceeds target threshold
CHK001  Checker cannot analyze operation
CHK002  Checker stage timed out
CHK003  Internal checker failure
```

Every diagnostic must contain, when available:

- Stable code
- Severity
- Stage
- Confirmed/inconclusive classification
- Human-readable message
- Candidate file and line
- Module and operation
- Scenario parameters
- Input/output shape evidence
- Original exception type and compact stack trace
- Reproduction command

Example report:

```json
{
  "report_schema_version": 1,
  "candidate_id": "candidate_0182",
  "candidate_hash": "sha256:...",
  "checker_version": "0.1.0",
  "overall_status": "FAIL",
  "gpu_submission_recommended": false,
  "diagnostics": [
    {
      "code": "SHP003",
      "severity": "error",
      "stage": "abstract_forward",
      "classification": "confirmed_candidate_failure",
      "message": "Residual operands have incompatible hidden dimensions",
      "file": "candidate/model.py",
      "line": 184,
      "scenario": {
        "batch_size": 32,
        "mode": "train"
      },
      "evidence": {
        "left_shape": [32, 512, 768],
        "right_shape": [32, 512, 1024]
      },
      "reproduction": "model-preflight check preflight.yaml --only abstract_forward"
    }
  ]
}
```

## 10. Policy Modes

Implement three modes:

### Audit

- Never recommends blocking a GPU submission.
- Records diagnostics and outcomes only.
- Use during historical replay and early deployment.

### Balanced

- Blocks confirmed candidate failures.
- Allows inconclusive candidates with a `GPU_CHECK_REQUIRED` marker.
- Recommended default for MLEvolve after shadow evaluation.

### Strict

- Blocks confirmed failures and inconclusive outcomes.
- Intended only for mature environments with high checker coverage.

The checker produces evidence and a recommendation. The caller remains responsible for the actual scheduling decision.

## 11. Detailed Implementation Phases

Complete phases in order. Do not begin parent-repository integration until the standalone CLI and schemas are stable.

### Phase 0: Failure taxonomy and evaluation corpus

Tasks:

1. Collect historical generated candidates that failed near training startup.
2. Separate confirmed bugs from timeouts, queue failures, environment failures, and jobs that entered training successfully.
3. Classify confirmed failures into source, construction, data, shape, loss, autograd, optimizer, validation, hardware, memory, and CUDA-only groups.
4. Create at least ten minimal valid/invalid fixtures representing high-frequency failures.
5. Finalize the initial diagnostic codes and status semantics.
6. Record expected outcomes for every fixture before implementing the checker.

Acceptance criteria:

- Every historical sample has a ground-truth category.
- Timeouts are not automatically labeled as buggy nodes.
- The initial diagnostic taxonomy covers the known corpus without using a generic error for most cases.

### Phase 1: Repository foundation and schemas

Tasks:

1. Initialize the package using a `src/` layout.
2. Configure `pytest`, Ruff, type checking, and build metadata.
3. Implement status, severity, diagnostic, stage-result, and final-report models.
4. Implement manifest loading, path normalization, validation, and JSON Schema.
5. Implement JSON and readable terminal reports.
6. Implement the CLI skeleton and stable exit codes.
7. Add golden tests for manifests and reports.

Required exit codes:

```text
0   PASS
10  Confirmed candidate failure
20  INCONCLUSIVE
30  Invalid manifest
40  Checker internal error
```

Acceptance criteria:

- Invalid manifests never execute candidate code.
- Reports validate against `schema_v1.json`.
- CLI exit codes match the documented status.
- Report serialization is deterministic.

### Phase 2: Isolated worker engine

Tasks:

1. Implement the controller/worker boundary.
2. Define a JSON-compatible worker request and response protocol.
3. Execute each runtime stage in a fresh subprocess.
4. Add wall-time, memory, process, stdout, and stderr limits.
5. Capture termination causes and distinguish timeout, signal, out-of-memory, candidate exception, and checker exception.
6. Ensure workers are cleaned up after success, failure, or cancellation.
7. Build the minimal CPU container.

Acceptance criteria:

- Candidate code is never imported into the controller process.
- A worker that hangs or spawns children is terminated cleanly.
- A timed-out worker returns `INCONCLUSIVE`, not `FAIL`.
- One candidate cannot affect a subsequent candidate.

### Phase 3: Static source checker

Implement an AST-based checker that executes no candidate code.

Checks:

- Python syntax
- Declared entry-point existence where statically resolvable
- Local import existence
- Direct `.cuda()` use
- Literal `device="cuda"`
- Model structure controlled by `torch.cuda.is_available()`
- Hard-coded batch sizes in `view`, `reshape`, and tensor factories
- Triton, FlashAttention, Transformer Engine, bitsandbytes, NCCL, DDP, MPS, and CUDA-stream usage
- Custom C++/CUDA extension compilation
- Obvious use of unsupported target precision in configuration

Rules:

- Emit a hard failure only when the defect is certain.
- Device-portability findings may be policy-controlled warnings or errors.
- Include source locations and suggested review actions.

Acceptance criteria:

- Static checks complete without importing the candidate.
- Every rule has positive and negative unit tests.
- No rule silently rewrites source code.

### Phase 4: Candidate adapter, import, and construction

Tasks:

1. Implement the `CandidateAdapter` protocol and function adapter.
2. Load candidate modules only inside workers.
3. Validate that the adapter can build a PyTorch model.
4. Validate that the model has expected trainable parameters.
5. Build the optimizer and verify that its parameters belong to the model.
6. Detect missing or empty optimizer parameter groups.
7. Support normal CPU construction and meta-device construction strategies.
8. Prevent unexpected network downloads in the default container.

Acceptance criteria:

- Import, construction, and optimizer failures produce distinct diagnostics.
- Large models can attempt meta construction without allocating full parameter storage.
- Unsupported meta construction falls back or becomes inconclusive rather than being mislabeled.

### Phase 5: Task fixture and data-contract validation

Tasks:

1. Define the task-fixture contract.
2. Require fixtures to pass through the real transform, tokenizer, dataset, and `collate_fn` path where available.
3. Validate batch container type, required keys, input ranks, input dtypes, target shapes, target dtypes, masks, and lengths.
4. Validate variable-length collation.
5. Validate partial final batches when `drop_last=False`.
6. Test empty, malformed, or corrupt fixture handling without confusing task-fixture errors with candidate errors.

Acceptance criteria:

- A representative batch reaches the candidate adapter exactly as a real training batch would.
- Dict/tuple mismatches, label dtype errors, mask mismatches, and collation failures are distinguished.
- The full task dataset is not required.

### Phase 6: CPU micro-training checker

Implement the highest-value real execution stage:

```python
model.train()
optimizer.zero_grad(set_to_none=True)

with torch.autograd.detect_anomaly(check_nan=True):
    loss = adapter.training_step(model, batch, context)
    validate_scalar_finite_loss(loss)
    loss.backward()

validate_required_gradients(model)
optimizer.step()
```

Checks:

- Training step returns a tensor.
- Loss is scalar or explicitly reducible according to the manifest.
- Loss is finite and requires gradient.
- Backward succeeds.
- Required trainable parameters receive gradients.
- Gradients are finite.
- Optimizer step succeeds.
- A second micro-step succeeds when stateful behavior is enabled.
- Validation/evaluation succeeds under `model.eval()` and `torch.no_grad()`.

Scenario requirements:

- At least one normal batch.
- Smallest batch the actual DataLoader may emit.
- Validation batch when configured.
- Do not force batch size one when the task contract guarantees `drop_last=True` or has a larger minimum batch.

Acceptance criteria:

- Detached losses and invalid backward paths are caught.
- Validation-only failures are caught.
- Non-finite values include the originating operation where PyTorch provides it.
- CPU timeout is inconclusive.

### Phase 7: Abstract execution and exact-shape matrix

Define an internal backend protocol:

```python
class AbstractExecutionBackend(Protocol):
    def supports(self, candidate, scenario) -> bool:
        ...

    def execute(self, candidate, scenario) -> "AbstractResult":
        ...
```

Implement backends in this order:

1. `torch.export` backend
2. Meta-device backend
3. Wrapped FakeTensor backend
4. Safe real-CPU fallback where configured

Tasks:

1. Build exact production-shape inputs without allocating equivalent activation data.
2. Test forward, output contract, and loss construction.
3. Test all five scheduler batch-size proposals.
4. Test boundary sequence or spatial dimensions declared by the task.
5. Test train/eval mode where abstract execution supports it.
6. Capture operation names, shapes, dtypes, module paths, and source locations.
7. Normalize known dimension errors into stable diagnostic codes.
8. Treat missing fake kernels, graph breaks, and unsupported Python behavior as inconclusive unless a real candidate error is independently confirmed.

Acceptance criteria:

- Matrix, convolution-channel, reshape, residual, concatenation, attention-mask, and output/loss mismatches are caught in mutation tests.
- Every scenario result identifies its batch size and boundary values.
- Export failure alone never becomes a confirmed candidate failure.

### Phase 8: Target-GPU profile checker

Implement a versioned data schema for hardware profiles.

Example:

```yaml
schema_version: 1
vendor: nvidia
name: V100-32GB
architecture: volta
compute_capability: "7.0"
vram_bytes: 34359738368
native_training_dtypes: [fp32, fp16]
unsupported_features:
  - bf16
  - tf32
  - fp8
  - nvfp4
  - transformer_engine
  - flash_attention_2
```

Tasks:

1. Create initial V100 and A10 profiles.
2. Validate requested precision against the profile.
3. Detect architecture-specific packages and custom-kernel requirements.
4. Check declared CUDA/PyTorch/library compatibility requirements.
5. Mark CUDA-only behaviors as requiring the real-GPU canary.
6. Keep profile rules inspectable and versioned rather than embedding them as scattered conditionals.

Acceptance criteria:

- BF16-only V100 candidates are rejected before GPU submission.
- GPU-specific operations that cannot be evaluated on CPU are clearly marked inconclusive.
- Hardware profiles validate against their schema.

### Phase 9: Resource estimation

Start with a transparent analytical estimate:

```text
estimated memory = parameters
                 + gradients
                 + optimizer states
                 + estimated saved activations
                 + configured safety margin
```

Tasks:

1. Calculate parameter memory from shapes and dtypes.
2. Calculate gradient memory according to trainable parameters.
3. Calculate optimizer-state memory for supported optimizers.
4. Estimate saved-activation memory from the abstract graph.
5. Report uncertainty sources: allocator fragmentation, tensor lifetime, cuDNN workspace, custom kernels, and checkpointing.
6. Make thresholds configurable.

Suggested initial policy:

```text
<70% target VRAM       low estimated risk
70-90%                 warning
>90%                   high-risk warning or policy-controlled failure
```

Acceptance criteria:

- Exact and estimated components are reported separately.
- The checker never claims that the estimate proves the absence of CUDA OOM.
- Memory estimation errors do not hide otherwise valid structural results.

### Phase 10: Reporting, caching, and reproducibility

Tasks:

1. Produce JSON, terminal text, and JUnit reports.
2. Include compact reproduction commands for each failed stage.
3. Cache deterministic completed results.
4. Use the following cache inputs:

```text
candidate source hash
+ normalized manifest
+ task fixture hash
+ checker version
+ PyTorch version
+ GPU profile version
+ policy
```

5. Do not cache internal errors, cancelled jobs, or incomplete timeouts as successful results.
6. Include environment and dependency metadata without recording secrets.

Acceptance criteria:

- Re-running an unchanged candidate returns an equivalent report.
- A relevant source, fixture, profile, dependency, or checker-version change invalidates the cache.
- Review agents can act on the report without parsing raw console logs.

### Phase 11: Evaluation and hardening

Build the mutation suite from valid baselines. Include at least:

- Incorrect `Linear.in_features`
- Incorrect convolution input channels
- Invalid residual addition
- Invalid concatenation
- Invalid reshape element count
- Hard-coded batch dimension
- Incorrect attention mask
- Incorrect output class count
- Wrong target dtype
- BCE/CrossEntropy contract mismatch
- Detached loss
- Loss without a trainable path
- Missing optimizer parameters
- Non-finite loss
- Non-finite gradient
- Validation-only failure
- DataLoader/collate mismatch
- CUDA-only dependency
- Unsupported V100 BF16 configuration

Add small valid models for:

- CNN classification
- Transformer encoder
- Autoregressive transformer
- UNet segmentation
- LSTM/GRU
- Conv1D audio or time series
- GNN
- Diffusion UNet
- Multi-input model
- Custom operator

Evaluate:

- Confirmed pre-GPU catch rate
- False hard-rejection rate
- Inconclusive rate
- Median and P95 validation latency
- Peak CPU memory
- Repair-agent success rate
- GPU submissions avoided
- GPU queue-hours avoided

Initial release goals, to be measured rather than assumed:

```text
>=90% catch rate for structural/data/loss first-batch failures
<2% false hard rejection
<60 seconds median validation time for ordinary candidates
0 timeouts classified as confirmed candidate bugs
100% final reports valid against the report schema
```

### Phase 12: Standalone release

Tasks:

1. Complete README quickstart and architecture documentation.
2. Document the manifest, adapter, diagnostic, profile, and report contracts.
3. Document checker limitations and the required target-GPU canary.
4. Establish semantic versioning.
5. Publish the first tagged release only after historical replay.
6. Pin supported Python/PyTorch combinations in a tested compatibility matrix.

Standalone release acceptance criteria:

- A clean checkout can install and run the examples.
- CPU container validation works without a visible GPU.
- CI runs unit, integration, mutation, golden-schema, and isolation tests.
- No dependency on MLEvolve or an agent framework exists.

## 12. MLEvolve Submodule Integration Plan

After the standalone release is stable, add it to MLEvolve:

```text
MLEvolve/
├── third_party/
│   └── model-preflight/       # pinned Git submodule
└── mlevolve/
    └── integrations/
        └── preflight.py       # MLEvolve-specific translation layer
```

The MLEvolve integration must:

1. Receive a generated model node.
2. Create or select the task fixture and adapter.
3. Generate a `preflight.yaml` manifest.
4. Invoke the submodule CLI in the CPU validation environment.
5. Store the JSON report with the node metadata.
6. Send confirmed diagnostics to the review agent.
7. Rerun preflight after each repair.
8. Limit automatic repair attempts, initially to three.
9. Admit `PASS` candidates to the scheduler.
10. Admit policy-approved `INCONCLUSIVE` candidates with `GPU_CHECK_REQUIRED` metadata.
11. Prevent `FAIL` candidates from creating GPU jobs.
12. Prevent `INTERNAL_ERROR` from being mislabeled as a model defect.

Recommended flow:

```text
generated candidate
    -> CPU preflight
    -> confirmed failure
    -> review-agent repair
    -> CPU preflight again
    -> scheduler admission
    -> target-GPU canary
    -> full training
```

The target-GPU canary must run inside the eventual full GPU job before expensive setup or full training. A successful canary should continue in the same allocation rather than submitting a second GPU job.

## 13. Optional CUDA Compilation Worker

Do not include the CUDA toolkit in the default small CPU image.

Later, add an optional `Dockerfile.cuda-build` worker invoked only when source inspection finds custom CUDA/C++ extensions. It may compile code for a declared architecture without running it, catching syntax, header, template, build-system, and target-code-generation failures.

Compilation success remains inconclusive regarding runtime CUDA correctness.

## 14. Code Quality Requirements

- Keep modules focused and interfaces typed.
- Use deterministic JSON output.
- Add tests with every diagnostic rule and checker stage.
- Avoid broad exception swallowing.
- Preserve original exceptions as structured evidence.
- Avoid global monkey-patching of PyTorch.
- Keep PyTorch-version compatibility logic in dedicated modules.
- Do not add unrelated functionality during implementation.
- Do not modify candidate code inside the checker.
- Document every public API and schema field.
- Use stable diagnostic codes even if human-readable wording changes.

## 15. Required Agent Progress Reports

During implementation, report after each phase:

1. Files added or changed.
2. Public interfaces introduced or modified.
3. Tests added and their results.
4. Known limitations or inconclusive cases.
5. Any deviation from this plan and the technical reason.
6. Next phase and remaining blockers.

Do not report a phase complete without running its relevant tests.

## 16. Final Definition of Done

The project is complete for initial MLEvolve adoption only when all of the following are true:

- The checker is a standalone installable repository.
- CLI, Python API, manifest schema, report schema, and exit codes are documented.
- Candidate runtime execution is isolated from the controller.
- Static, data-contract, abstract-forward, CPU-training, validation, hardware-profile, and resource-risk stages exist.
- All five MLEvolve batch-size proposals can be tested abstractly.
- Confirmed failure, inconclusive analysis, timeout, and checker failure remain distinct.
- The mutation suite passes.
- Historical MLEvolve failures have been replayed and measured.
- False rejection and catch-rate metrics are reported.
- V100 and A10 profiles exist and reject known precision incompatibilities.
- The CPU container requires no GPU.
- The checker does not claim GPU emulation or replace the target-GPU canary.
- A tagged standalone release is ready to pin as an MLEvolve Git submodule.

