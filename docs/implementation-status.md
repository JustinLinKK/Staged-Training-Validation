# Implementation status against the plan

The v0.2.1 research release includes the standalone checker, opt-in reduction, parallel CPU
scenarios, and speculative integration. This file keeps release-gate claims separate from
implemented capabilities. See the [paper-release guide](paper-release.md) for evidence scope.

| Plan phase | Status | Evidence / limitation |
|---|---|---|
| 0: taxonomy and corpus | Partial | The original 12 mutations, expanded synthetic audits, and architecture controls are retained. The initial comparison also inspected nine archived candidates with uncertain labels/synthetic fixtures; this is not real-data historical validation. |
| 1: foundation and schemas | Implemented | Installable `src` package, manifest/report/profile schemas, CLI/API, deterministic serialization, exit codes, Ruff, mypy, pytest. |
| 2: isolated workers | Implemented | Fresh process per runtime stage, JSON protocol, process groups, time/CPU/address-space/PID/output limits, captured logs, offline environment, non-root CPU container. External container controls remain required for hostile native code. |
| 3: static checks | Implemented | Syntax, entry/relative import resolution, CUDA devices/branches/streams, batch literals, distributed/GPU libraries, and extension compilation findings. |
| 4: adapter and construction | Implemented | Protocol/function adapter, worker-only import, model/parameter/optimizer ownership validation, meta construction. |
| 5: task/data contract | Implemented | Real adapter batch path, recursive batch evidence, explicit input-rank and target-dtype checks, normal/final batch scenarios. Task-specific masks/lengths remain the adapter/loss contract unless an integration adds explicit fields. |
| 6: CPU micro-training | Implemented | Finite scalar loss, output class contract, anomaly-enabled backward, required and finite gradients, optimizer step, optional second step, validation path. |
| 7: abstract execution | Implemented | `torch.export`, meta, wrapped FakeTensor, optional bounded real-CPU fallback, full scenario matrix, operation error normalization, activation evidence. |
| 8: GPU profiles | Implemented | Versioned V100-16/32, A10-24, and A100-40/80 profiles with dtype rejection and GPU-canary markers. |
| 9: resource estimate | Implemented | Exact parameter/gradient bytes and labeled optimizer/activation/margin estimates with uncertainty. |
| 10: reports/cache | Implemented | JSON/text/JUnit, reproduction commands, content-addressed opt-in cache keyed by candidate, fixture, manifest, checker, PyTorch, profile, and selected stages. |
| 11: evaluation/hardening | Partial | Unit, integration, isolation, and mutation tests plus a historical-corpus evaluator. The broader model zoo, real historical measurements, and GPU queue-hour metrics require external artifacts/infrastructure. |
| 12: release | Research snapshot | v0.2.1 has repository/wheel installation, a versioned documentation wiki, citation metadata, and unchanged Docker options. This paper software release does not assert that the original acceptance gates passed. No public PyPI release is intended. |
| v2: shape reduction | Experimental | Original-shape gates, supported hidden-width reduction, numerical guards, and full-width fallback are implemented. Reduction remains disabled by default; original paired gates failed, and later auto-only audits have narrower scope. |
| v2: concurrency | Implemented | Configurable scenario workers and CPU threads retain isolated processes and deterministic report ordering. The synthetic concurrency comparison measured 1.78x median paired speedup with four workers. |
| v2: speculative integration | Implemented | `submit()` and JSONL events expose provisional static admission followed by the full configured workflow; queue actions remain framework-owned. |

The `model_preflight` package does not implement a GPU scheduler, auto-repair, GPU emulation,
or a target-GPU canary. The separate root compatibility plugin offers scheduler callbacks;
the calling framework still owns job submission and cancellation.
