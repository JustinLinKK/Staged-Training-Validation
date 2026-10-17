# Width scaling and numerical fallback — 2026-10-16

**The original models had too little reducible computation to cover the analysis cost.**
Larger CNN/residual models demonstrate end-to-end acceleration, but width alone is not a
reliable eligibility rule. The three missed runs were one overflow defect at three seeds;
revision 3 now detects all three through original-width fallback.

## What changed

The reducer inspects the traced training and validation workflows for explicit operations
with overflow or singularity risks: exponentials, logarithms, powers, division/reciprocal,
and square roots. If present, it retains the original model and executes the original CPU
workflow. Both tensor methods and function-call forms are covered. Evidence records
`numerically_sensitive_operation:<operation>`. The reducer revision is now 3, separating
cached results from revision 2. The candidate loss and parameters are not rewritten.

This conservative guard fixes the observed defect class; it does not prove numerical
equivalence for arbitrary code. Sensitive math inside other operations, value-dependent
matrix overflow, and other width-dependent failures can still require original-width
numerical checks. Automatic reduction remains opt-in and disabled by default.

## Detection replay

The original overflow model produces logits of 128 at full width and 64 when reduced.
Its exponential loss overflows only at full width. The new guard retains full width:

| Seed | Original-width baseline | Revision 2 auto | Revision 3 auto |
| --- | --- | --- | --- |
| 11 | FAIL, NUM001 | PASS (missed) | FAIL, NUM001 |
| 23 | FAIL, NUM001 | PASS (missed) | FAIL, NUM001 |
| 37 | FAIL, NUM001 | PASS (missed) | FAIL, NUM001 |

The fresh focused replay covered all eight workflow mutations and all six adversarial
cases, three seeds each, both off and auto: **84 checks / 42 pairs**. Both arms detected
**33/33 labeled defect runs** with the expected diagnostic codes. Both had zero false hard
rejections among nine control runs, zero checker errors, and zero inconclusive outcomes.
Status and diagnostic-code agreement was 42/42. The square-root gradient defect also now
uses direct fallback, preserving detection while avoiding a failed reduced attempt/retry.

This is a focused regression replay, not a rerun of the original 48-case acceptance corpus.
The quality output retains the full 39-case synthetic inventory and deliberately reports
`complete: false` because only the 14 planned cases were selected. `run_plan.json` records
that selection before measurement. The original revision-2 evidence remains unchanged.

## Does increasing model size help?

The separate size study used **12 valid workloads, five paired repetitions each: 120 checks**.
It retained the previous largest size in each family as a fresh control, then added two
larger widths. Only hidden width changed within each family; depth, batch size, inputs,
image resolution, optimizer, loss, precision, and training-step count stayed fixed.

All 120 checks passed. All 180 automatic CPU scenarios applied reduction, with no retries.
Ratios are medians of paired off/auto times, so they need not equal the ratio of the two
separate latency medians. Values above one are faster. Intervals use 2,000 fixed-seed
bootstrap resamples of five pairs.

| Model / width | Off median s | Auto median s | Paired speedup [95% CI] |
| --- | ---: | ---: | --- |
| MLP 2,048 | 7.446 | 7.963 | 0.934 [0.920, 0.961] |
| MLP 8,192 | 10.389 | 10.223 | 1.024 [0.945, 1.061] |
| MLP 16,384 | 17.753 | 16.008 | 1.129 [1.089, 1.166] |
| CNN 128 | 8.483 | 8.650 | 0.962 [0.917, 1.023] |
| CNN 256 | 9.056 | 8.769 | 1.041 [1.005, 1.079] |
| CNN 512 | 12.245 | 9.694 | **1.262 [1.252, 1.287]** |
| Residual 128 | 8.330 | 8.495 | 0.990 [0.920, 1.024] |
| Residual 256 | 10.279 | 9.066 | 1.130 [1.035, 1.146] |
| Residual 512 | 16.836 | 11.008 | **1.535 [1.469, 1.548]** |
| Image/tabular 128 | 8.136 | 8.607 | 0.946 [0.883, 0.988] |
| Image/tabular 256 | 8.487 | 9.098 | 0.939 [0.862, 0.964] |
| Image/tabular 512 | 8.492 | 9.212 | 0.950 [0.899, 0.985] |

![Width scaling](../.cache/shape-reduction-v3-scale/scaling.png)

The original MLP-2048 workflow consumed only about 0.147 seconds of an 8.286-second check.
Even removing all of that computation could save only about 1.8%, before adding reducer
overhead. The new data shows why larger work can change the outcome:

- Wide-to-wide convolutions gain substantial work as width grows, while each parameter
  is reused over spatial positions. CNN-512 and residual-512 exceed 1.25x individually.
- MLP-16384 achieves a 5.042x inner-workflow speedup but only 1.129x end to end. Its median
  reduction analysis takes 4.416 seconds across CPU workers, versus 1.207 seconds at width
  2,048. Original parameter initialization, scanning, slicing, and checking remain costly.
- The image/tabular model connects widened layers to fixed dimensions (3/32 image channels,
  12/16 tabular features, and a 48-feature concatenation boundary). Its computation grows
  roughly linearly with width rather than like the wide-to-wide CNN layers. Its inner
  workflow becomes faster, but the whole check remains slower at every tested width.

These results support selecting reduction based on expected computation saved relative to
analysis/setup cost, rather than parameter count or width alone. No such performance-selection
heuristic was added in this patch. This timing-only study does not re-evaluate the original
aggregate acceptance gates; the summary marks them as unevaluated (`null`).

## Controls and verification

The corpus was frozen before measurement. Seeds were 11/23/37 for quality and
101/103/107/109/113 for timing. Runs were sequential, arm order alternated, caching was
disabled, CPU FP32 was used, and OMP/MKL threads were one. Synthetic fixtures, 30-second
abstract/60-second CPU timeouts, and the 8-GiB worker memory limit were unchanged. MLP batches
had 64 samples; image batches had eight 3x64x64 images; final batches had one sample.

The same AMD Ryzen 9 9950X3D WSL2 host and Python 3.13/PyTorch 2.11 environment were used.
Unrelated workloads were preserved. Timing-trial starting one-minute load ranged from
1.09 to 3.38, median 1.59. Fresh small-size controls avoid treating the older campaign's
different contention as a size effect. Five repetitions and a shared host still limit
precision; these are synthetic CPU workflow results, not GPU or real-data training results.

Measured checker SHA-256:
`33898c7214c6ee18b9bbd2c4305a9cbd7445b8a16c189859e460d333fa2840a7`.
Only analysis/plot generation was expanded after freeze; measured checker code stayed fixed.

- **68 tests passed**, including new guards for tensor/function syntax and validation math.
- Mypy passed all 47 source files; sdist and wheel builds passed.
- Changed Python files pass Ruff; full Ruff retains the pre-existing E501 at
  `src/model_preflight/checks/source/ast_checker.py:357`.
- All **204 report schemas** validate. All **102 pairs** retain identical original
  prerequisite evidence and memory estimates; reduced runs retain the required GPU-check flag.

## Artifacts and reproduction

[Quality artifacts](../.cache/shape-reduction-v3-quality/) and
[size-study artifacts](../.cache/shape-reduction-v3-scale/) include frozen sources/manifests,
raw reports, measurements, CSV/JSON summaries, audits, and the predeclared run plan.
Plots are available as [PNG](../.cache/shape-reduction-v3-scale/scaling.png),
[SVG](../.cache/shape-reduction-v3-scale/scaling.svg), and
[PDF](../.cache/shape-reduction-v3-scale/scaling.pdf).

Run from the repository root with the model-preflight environment; use fresh output paths
to repeat measurements rather than resume existing ones:

```bash
PYTHONPATH=src:. python scripts/run_shape_reduction_experiment.py freeze --output .cache/v3-quality-repeat
PYTHONPATH=src:. python scripts/run_shape_reduction_experiment.py run \
  --output .cache/v3-quality-repeat --phases quality \
  --cases detached_loss,vector_loss,nonfinite_loss,nonfinite_gradient,second_backward,width_dependent_detach,missing_gradient,validation_failure,optimizer_step,optimizer_shape_assumption,tail_channel_nan,reshape_constant,shared_parameters,width_dependent_overflow
PYTHONPATH=src:. python scripts/run_shape_reduction_experiment.py freeze \
  --output .cache/v3-scale-repeat --scale-study
PYTHONPATH=src:. python scripts/run_shape_reduction_experiment.py run \
  --output .cache/v3-scale-repeat --phases timing
PYTHONPATH=src:. /home/justin/miniconda3/envs/perfseer/bin/python \
  scripts/run_shape_reduction_experiment.py summarize --output .cache/v3-scale-repeat --plots
```

The last command uses an existing environment containing Matplotlib; no dependencies were added.
