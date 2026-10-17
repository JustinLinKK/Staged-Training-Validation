# Manifest v1

The authoritative contract is `model_preflight/manifest/schema_v1.json`. YAML is accepted as the
wire format. Unknown fields are rejected to catch misspellings.

`candidate.root` and `task.fixture_root` resolve against the manifest's directory. `adapter` is a
required `module:attribute` entry point. `task.input_rank`, when set, excludes the batch
dimension. `task.target_dtype` enables an explicit target check rather than relying on loss
heuristics. Set `task.output_class_dimension` (often `-1` for classification/transformers or
`1` for segmentation) to enforce `num_classes` on the captured model output.

`train_batch_sizes` should contain every scheduler proposal. Abstract execution forms the cross
product of batch sizes, normal and boundary input shapes, and requested precisions. Real CPU
training uses the first normal proposal plus `last_batch_size` when `test_last_batch` is true.
The last-batch scenario is omitted when `task.drop_last` is true.

Abstract execution remains meta-only by default. `allow_real_cpu_abstract_fallback` permits a
resource-bounded real CPU forward when an operation has no meta implementation.

Built-in profile names omit `.yaml`, for example `nvidia/v100_32gb`. A relative `.yaml` path
selects a custom profile relative to the manifest.

`memory_failure_fraction` is optional. Crossing it remains an inconclusive estimate—not a
confirmed defect—but recommends blocking in balanced/strict modes.

## Parallel execution

```yaml
execution:
  max_workers: 4
  cpu_threads: 1
```

Both settings are positive integers and default to `1`, retaining serial execution.
`max_workers` bounds concurrent scenarios within the abstract-forward and CPU-training stages.
Controller threads launch fresh worker processes; candidate code stays isolated in those
processes. Construction, data-contract, and the entire original-shape abstract matrix still
finish before reduced CPU workflows start. Validation remains a separate stage. Results and
diagnostics retain manifest order, and each reduced scenario's original-width retry occupies
the same worker slot. A timeout or candidate failure does not discard other scenario results.

`cpu_threads` sets PyTorch intra-operation threads and the OpenMP, MKL, OpenBLAS, NumExpr, and
Accelerate thread environment limits. PyTorch inter-operation threads remain `1`. Use more
worker slots for many scenarios, or more CPU threads for a few computationally large scenarios.
Keep `max_workers * cpu_threads` within the CPU capacity allocated to this check, accounting for
other checks running on the host. Candidate-created thread pools remain candidate-controlled.

Memory and process limits still apply per worker; parallel workers can together use up to
`max_workers * maximum_cpu_memory_mb` MiB of address space. POSIX process-count limits may also
count other processes/threads owned by the same user. Choose concurrency within host/container
memory and PID limits. Each worker retains its own temporary directory, but adapters that write
shared files outside it should use serial execution.

On Linux, the default `maximum_processes: 32` can prevent extra CPU threads on a busy user
account even when CPU and RAM are available. The checker reports an exhausted thread-start
budget as `INCONCLUSIVE` (`CHK001`) before importing the candidate. To use `cpu_threads: 2`,
set an explicit process limit with room for that user's existing tasks and the worker pools
(the local benchmark uses `maximum_processes: 4096`); the checker does not raise this limit
automatically.

Resource limits are installed inside each fresh worker before PyTorch and candidate imports;
launching from controller threads does not use `preexec_fn`. The CPU-time allowance scales with
`cpu_threads`, since POSIX CPU time sums time across threads. The wall-time timeout is unchanged
and starts when the worker launches, excluding time queued for a slot.

For parallel stages, `duration_seconds` measures elapsed wall time, `evidence.worker_seconds`
retains summed worker durations including retries, and `evidence.parallel_workers` records the
slot limit. Individual scenario durations and `cpu_threads` remain in worker evidence. Execution
settings are included in the normalized manifest and cache identity. A fixed `execution.seed`
keeps worker RNG initialization independent of scheduling; changing numerical thread counts can
still change floating-point reduction order.

The [concurrency benchmark](../scripts/benchmark_parallel_execution.py) compares serial v2,
four workers with one CPU thread each, and two workers with two CPU threads each. All arms use
automatic shape reduction, four original-shape abstract scenarios, normal/final training
batches, separate validation, identical limits, and disabled caching. It rotates arm order over
three repetitions with seeds 11, 23, and 37. The default synthetic cases are a wide MLP, CNN, and
Transformer encoder; `--cases` can select existing mutation or architecture controls.

```bash
PYTHONPATH=src:. python scripts/benchmark_parallel_execution.py \
  --output .cache/parallel-execution-comparison
```

Use a new output directory for each run. It saves the implementation and candidate sources,
manifests, raw reports, host load, per-run measurements, CSV, and a JSON summary. Parity checks
compare diagnostics, stage outcomes, original-model memory estimates, reduction decisions, and
exact seeded training losses between serial and four-worker execution. The two-thread arm may
differ numerically due to floating-point accumulation order. Peak RSS is per worker, not total
memory used by all concurrent processes. This comparison measures concurrency with v2 enabled;
it does not rerun or replace the separate shape-reduction acceptance experiment.

Measured on an AMD Ryzen 9 9950X3D (32 logical CPUs), Python 3.13.12, and PyTorch 2.11.0:

| Synthetic workload | Serial median | 4 workers × 1 thread | 2 workers × 2 threads | Paired speedup, 4 workers |
|---|---:|---:|---:|---:|
| MLP, width 2048 | 12.54 s | 7.28 s | 8.38 s | 1.72× |
| CNN, width 128 | 12.69 s | 7.13 s | 8.53 s | 1.78× |
| Transformer encoder | 13.09 s | 7.18 s | 8.68 s | 1.82× |

All 27 checks passed with matching diagnostic/stage outcomes, memory estimates, and reduction
decisions. All 81 CPU scenarios applied reduction. Training losses matched exactly between
serial and four-worker runs for every paired seed. Across the nine pairs, the median end-to-end
speedup was **1.78×** with four workers and **1.50×** with two workers/two threads. Four workers
improved median paired abstract-stage latency by **3.12×** and CPU-training latency by **1.85×**.
Host one-minute load ranged from 0.96 to 1.90; existing host workloads were left running.

The [raw timing reports](../.cache/parallel-execution-20261016/raw),
[CSV](../.cache/parallel-execution-20261016/measurements.csv), and
[summary](../.cache/parallel-execution-20261016/summary.json) are local artifacts excluded from
Git. Three repetitions on three synthetic models establish a useful local improvement, not a
universal speedup or a new result for the original eight-workload reduction acceptance gate.

A separate nine-run replay (one fixed seed, all three configurations) detected detached loss,
second-step optimizer failure, and validation failure with matching diagnostics and fallback
decisions in every run. Its [reports](../.cache/parallel-defects-20261016/raw) and
[summary](../.cache/parallel-defects-20261016/summary.json) are also local artifacts. Reproduce it
with `--repetitions 1 --cases detached_loss audit_optimizer_step_2 validation_failure` and a new
output directory. The full test suite passed 217 tests with six optional dependency/image skips;
Ruff, mypy, and sdist/wheel builds passed.
