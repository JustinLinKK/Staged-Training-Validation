# Architecture

## Stage order

| Stage | Evidence |
|---|---|
| `static_source` | Syntax, adapter entry points, device/batch literals, distributed APIs, GPU-only imports, and custom extensions |
| `hardware` | Requested precision against the selected versioned target profile |
| `construction` | Original model construction, parameters, optimizer type, and parameter ownership |
| `data_contract` | Real adapter fixture/batch path, ranks, and target dtype |
| `abstract_forward` | Every original-shape batch/shape/precision scenario using export and meta execution |
| `cpu_training` | Finite scalar loss, backward, gradients, optimizer updates, optional second step, and normal/final batches |
| `validation` | Separate configured validation under `eval()` and `no_grad()` |
| `memory` | Original-model parameter/gradient bytes and labeled optimizer/activation estimates |

Only eligible CPU training/validation workers can apply [shape reduction](shape-reduction.md),
after original construction, data, and abstract checks pass. [Parallel execution](manifest.md#parallel-execution)
keeps stage dependencies while overlapping independent scenarios. [Speculative submission](agent-integration.md)
returns source-check evidence first and continues this complete pipeline in the background.

## Isolation and evidence

The controller loads and normalizes the manifest, hashes inputs, runs AST and target-profile
checks, dispatches isolated workers, applies policy, and serializes the report. It never imports
candidate modules.

Each construction, data, abstract, training, or validation request starts
`model_preflight.execution.worker_main` in a new process. Requests and responses use protocol
version 1 and JSON-compatible values. Candidate stdout/stderr are redirected before import.
Timeouts and resource-limit terminations become `INCONCLUSIVE`; malformed worker responses and
unexpected worker crashes become `INTERNAL_ERROR`.

The PyTorch abstract path first attempts `torch.export` when a conventional input can be
identified. Regardless of export success, meta execution checks the adapter's training path and
collects leaf-module output sizes. Unsupported abstract semantics are inconclusive. Known,
unambiguous dimension errors are confirmed failures.

The framework-facing boundaries are the candidate adapter protocol and the worker stage
handlers. A future backend can implement the same adapter/abstract-result contracts without
changing report policy.

The analytical memory estimate separates exact parameter/gradient storage from estimated
optimizer state, saved activations, and a safety margin. It always records uncertainty and never
asserts that CUDA OOM is impossible.

