# Preflight v2 comparison — 2026-10-16

For release scope and availability of raw artifacts, see the [paper-release guide](paper-release.md).

The implementation works as an opt-in reduced-width workflow checker, but **both acceptance
gates failed** on the frozen corpus. Keep `execution.shape_reduction.mode: off` as the default.
The experiment supports the structural separation, but does not establish defect-detection
parity or an end-to-end speed improvement.

| Acceptance check | Measured result | Outcome |
| --- | --- | --- |
| No additional missed defects | Baseline 69/69; auto 66/69 labeled defect runs | Fail |
| No additional false hard rejections or checker errors | Zero of either | Pass |
| Median end-to-end speedup at least 1.25× | 0.934×, 95% bootstrap interval 0.925–0.965× | Fail |
| Aggregate replay P95 slowdown at most 10% | 19.435 → 19.851 s, +2.14% | Pass |
| Actual reduction on all eight supported workloads | 8/8; all 120 timing CPU scenarios reduced | Pass |

The overall quality and performance gates are conjunctions: passing individual checks does
not compensate for the failures above. The observed speed ratio means about 7.1% longer
end-to-end runtime for automatic reduction under the specified aggregate statistic.

## Implementation boundary

Source, hardware, construction, data-contract, and the complete abstract scenario matrix
retain original dimensions. Only subsequent CPU training/validation workers may reduce
eligible dimensions. Original construction and abstract evidence determine the memory estimate.

The FX reducer couples producers, consumers, normalization, flattening, and compatible
residuals. It pins external interfaces and concatenation boundaries, preserves candidate
methods and optimizer configuration, and builds the real optimizer after slicing parameters.
Disposable meta probes compare training/validation operation structure and output contracts.
Unsupported or uncertain cases fall back. A failed reduced execution receives one fresh
original-width retry; the original result determines candidate failure and both attempts count
toward latency. Successful reduced evidence requires later original-width numerical validation.

See [configuration and implementation details](shape-reduction.md).

## Frozen experiment

The run contains **416 reports**: 48 v1 references, 288 quality runs, and 80 dedicated timing
runs. Corpus sources, expected labels, seeds, and thresholds were frozen before measurement.

| Corpus | Unique cases | Quality runs per arm | Baseline / auto detected defects |
| --- | ---: | ---: | ---: |
| Existing mutations | 12 | 36 | 36/36 / 36/36 |
| Valid performance workloads | 8 | 24 | Not defect-labeled |
| Workflow mutations | 8 | 24 | 24/24 / 24/24 |
| Adversarial controls and defects | 6 | 18 | 9/9 / 6/9 |
| Unsupported controls | 5 | 15 | Not defect-labeled |
| Historical candidates | 9 | 27 | Labels unknown; excluded from recall |

Quality seeds were 11, 23, and 37. Timing seeds were 101, 103, 107, 109, and 113, with five
pairs per workload and alternating arm order. Runs were sequential, cache was disabled,
CUDA was hidden, and OMP/MKL thread counts were one. Fixtures, scenarios, precision,
timeouts, and step counts were identical within each pair. Synthetic workloads used FP32,
normal/final batches, separate validation, and two training steps except the explicitly
single-step overflow adversary. Archived manifests retained their original scenario settings.

The eight workloads were MLP widths 512/2048, CNN and residual widths 64/128, and image/tabular
widths 64/128. MLP inputs had 128 features and batch size 64; image inputs were 3×64×64 with
batch size 8. Multimodal models also had 12 tabular features. Last batches had one sample.
Inputs, resolutions, targets, and layer counts were not reduced.

The host was an AMD Ryzen 9 9950X3D under WSL2 with 32 available logical CPUs, Python 3.13.12,
PyTorch 2.11.0, and approximately 64 GiB RAM. Existing unrelated workloads were preserved.
One-minute host load at measured trial starts ranged from 1.60 to 6.02, median 2.94; this was
not an exclusive-host benchmark. Per-trial load snapshots and package versions are retained.

Reference commit: `57f199c5446ba5c36f509fd8a808e67848853919`.
Measured checker source SHA-256: `e2ce3db2873ad23c63ccb42909fa38267bd5d24defaa272cf3a03aab71e2deb1`.
Reducer revision: 2. Forty-seven unchanged v1 reference results were reused from the earlier
campaign after verifying identical source bytes, equivalent manifests, profile bytes, and
reference commit; the additional cached-model reference was run fresh. Reference timings are
not used for speedup. Every final off/auto quality and timing measurement was fresh on revision 2.
Earlier development measurements are excluded. Analysis-only aggregation was expanded after
freeze; measured checker code, corpus, labels, and thresholds were unchanged.

## Quality result and numerical counterexample

Reduction-off reproduced v1 status, diagnostic codes, and classifications for all 48 cases.
Across the 144 quality pairs, off/auto status and diagnostic-code agreement was 141/144
(97.92%). Synthetic defect recall fell from 100% to 95.65%: one of 23 defect cases was missed
at all three seeds. Both arms had zero false hard rejections among 48 labeled nondefect runs,
zero checker-error diagnostics, and zero overall internal errors.

The counterexample has two linear layers, hidden width 128, and an exponential loss. Its
initial parameters produce logits of 128 at original width and 64 after slicing. In FP32,
the full-width exponential overflows while the reduced loss and gradients remain finite.
Both models have valid shapes and the same operation structure. The baseline reports
`NUM001`; automatic reduction passes. A retry on reduced failure cannot recover this case,
because the reduced execution succeeds. Marking `gpu_check_required` makes the evidence
boundary explicit but does not satisfy the defect-recall gate.

Width-dependent detach behavior, optimizer configuration changes, tail-channel NaNs, constant
reshapes, and shared parameters retained baseline outcomes through conservative fallback.
All eight workflow mutations retained their expected diagnoses, including missing/nonfinite
gradients, optimizer-step failure, and second-step state failure.

Inconclusive outcomes were unchanged: three synthetic runs (dynamic control flow) and twelve
historical runs per arm. They are not counted as detected defects.

## Performance result

Ratios below are paired baseline/auto medians; values above one are faster. End-to-end
intervals use 2,000 bootstrap resamples of five pairs. The aggregate interval resamples the
eight workload medians. These small-sample intervals describe this fixed workload/host
experiment, not a population guarantee.

| Workload | Off median s | Auto median s | End-to-end ratio [95% CI] | CPU-stage ratio | Inner workflow ratio |
| --- | ---: | ---: | --- | ---: | ---: |
| MLP 512 | 8.031 | 8.637 | 0.929 [0.876, 0.954] | 0.871 | 1.441 |
| MLP 2048 | 8.286 | 8.917 | 0.929 [0.917, 0.965] | 0.858 | 3.124 |
| CNN 64 | 8.174 | 8.562 | 0.965 [0.924, 0.992] | 0.885 | 2.180 |
| CNN 128 | 8.577 | 9.080 | 0.952 [0.884, 1.013] | 0.902 | 2.642 |
| Residual 64 | 8.490 | 9.081 | 0.925 [0.900, 1.011] | 0.856 | 1.968 |
| Residual 128 | 8.740 | 9.042 | 0.974 [0.930, 1.005] | 0.963 | 2.550 |
| Image/tabular 64 | 7.874 | 8.505 | 0.923 [0.911, 0.929] | 0.864 | 1.336 |
| Image/tabular 128 | 8.374 | 8.836 | 0.938 [0.921, 0.975] | 0.883 | 1.575 |

The inner workflow benefits from fewer operations, but reduction analysis takes a median
1.21–1.34 seconds summed across the three CPU workers per check. Original-shape checking,
model construction, and worker startup remain. None of the eight workload medians improves
end-to-end or CPU-stage latency. Reduction timing includes its probe construction and optimizer
analysis; it should not be interpreted as a separately additive net overhead estimate.

Peak worker RSS improves for seven workloads, but not uniformly: residual 128 falls from
836.9 to 741.7 MiB, while MLP 512 changes from 636.5 to 636.7 MiB. Workers initially allocate
the original model, so reduced parameter counts do not imply proportional peak-memory savings.
RSS is the maximum observed worker self peak, not aggregate concurrent memory or descendant RSS.

| Quality replay group | Off median / P95 s | Auto median / P95 s |
| --- | ---: | ---: |
| Synthetic | 7.864 / 8.552 | 8.443 / 11.618 |
| Historical | 18.626 / 20.949 | 19.236 / 21.239 |
| Combined | 7.946 / 19.435 | 8.628 / 19.851 |

Combined P95 meets the 10% limit, but synthetic P95 worsens by **35.9%**, partly from 36
original-width confirmation retries. Slower historical cases dominate the combined P95.
Per-case median/P95 end-to-end, CPU-stage, workflow, reduction, and memory statistics are in
the CSV/JSON breakdown; regressions and exclusions are retained rather than averaged away.

## Eligibility and historical applicability

Across automatic quality runs, 285 scenario reduction analyses were attempted: 135 applied
(47.4%) and 150 fell back (52.6%). Applied counts include the 36 failed reduced attempts that
received original-width retries. Eight supported performance cases reduced in every quality
and timing scenario, with no timing retries. Other scenarios that failed prerequisites did not
attempt reduction. Evidence records each fallback reason separately.

The nine archived sources were matched to immutable prior reports by source-tree hash.
Two duplicate repaired sources were excluded (`conservative_hwdb_on/repaired.py` and
`normal_hwdb_off/repaired.py`). No included replay failed for a missing dependency or fixture.
All nine historical labels remain uncertain and are excluded from recall and false-rejection
calculations. Their fixtures are synthetic image/tabular tensors (3×256×256 and 12 features),
not real-data validation. `timm` is absent from the environment but was not required by these
included execution paths; installed package versions and the local diagnostics-helper hash
are recorded with the artifacts.

Four archived cases reached reduction analysis: three had no reducible hidden dimensions
and one used unsupported `float()` behavior. Five failed the original prerequisites. Thus
actual historical application is **0/9 cases**, with 27/27 paired diagnostic agreement.
This archive sample provides no evidence of real-candidate acceleration.

## Verification, artifacts, and reproduction

The independent report audit validates all 416 report schemas. All 184 quality/timing pairs
have identical original prerequisite evidence (excluding timing/RSS/log text) and resource
estimates. Reduced results consistently require original-width numerical validation. No
recovered internal errors were found in nested attempts.

Unit coverage includes dimension coupling, interfaces, parameter values/names/trainability,
optimizer ownership, residual/concat behavior, probe isolation, fallback attribution, cache
separation, prerequisites, memory provenance, `--only`, and seeded execution. The experiment
additionally executes second-step and validation defects through fresh worker processes.
Final verification receipts are saved with the run. Full Ruff has one pre-existing E501 at
`src/model_preflight/checks/source/ast_checker.py:357`; changed Python files pass Ruff.
The final full suite passes 61 tests; mypy passes all 47 source files, and both the source
distribution and wheel build successfully. `git diff --check` passes.

Artifacts are under [`../.cache/shape-reduction-v2-20261016-final/`](../.cache/shape-reduction-v2-20261016-final/):

- [Summary JSON](../.cache/shape-reduction-v2-20261016-final/summary.json),
  [per-case/category CSV](../.cache/shape-reduction-v2-20261016-final/breakdown.csv), and
  [per-trial CSV](../.cache/shape-reduction-v2-20261016-final/measurements.csv).
- [PNG](figures/comparison.png),
  [SVG](figures/comparison.svg), and
  [PDF](figures/comparison.pdf) comparison plots.
- [Frozen corpus](../.cache/shape-reduction-v2-20261016-final/corpus.json), raw manifests/reports/logs
  in `raw/`, frozen checker sources in `implementation/`, and original v1 sources in `baseline/`.
- [Audit](../.cache/shape-reduction-v2-20261016-final/audit.json),
  [verification](../.cache/shape-reduction-v2-20261016-final/verification.json), host/dependency
  receipts, and `reference_reuse.json`. Keep the earlier reference directory named in that receipt.

From the repository root, with v1 commit above as HEAD and this implementation in the working
tree, create a new output directory to repeat the full comparison:

```bash
PYTHONPATH=src:. /home/justin/miniconda3/envs/model-preflight/bin/python \
  scripts/run_shape_reduction_experiment.py freeze --output .cache/shape-reduction-v2-repeat \
  --archive-root /home/justin/MLEvolve/records/2026-09-14_petfinder_workflow_matrix_final
PYTHONPATH=src:. /home/justin/miniconda3/envs/model-preflight/bin/python \
  scripts/run_shape_reduction_experiment.py run --output .cache/shape-reduction-v2-repeat
PYTHONPATH=src:. /home/justin/miniconda3/envs/perfseer/bin/python \
  scripts/run_shape_reduction_experiment.py summarize \
  --output .cache/shape-reduction-v2-repeat --plots
```

The plotting environment supplies Matplotlib; measurement dependencies were not changed.
The harness resumes completed measurements, so use a fresh directory for an independent run.
Re-summarizing the delivered directory regenerates its CSV/JSON and plots without rerunning models.

Further work would need to reduce analysis overhead and address numerical coverage before
default enablement. Reusing validated plans or choosing only compute-heavy workloads may help
latency, but cannot by themselves recover width-dependent numerical defects. A full-width
numerical check would require its own cost/coverage experiment. None of those unmeasured
changes is claimed as a solution here.
