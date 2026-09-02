# TinyStories W4A8 screening accuracy

## Result

The `abs(W4 code) >= 4` threshold used to stress the scheduler is not an
accuracy-safe pruning policy. On the same 32-story, 2,016-token official
validation slice used for W4A8 QAT evaluation, it is catastrophic even when
all activations are retained.

The direct baseline is the dense 500-step W4A8 QAT model:

- original FP32 perplexity: 2.361222;
- dense W4A8 QAT perplexity: 2.710342;
- dense W4A8 top-1 agreement with FP32: 86.3%.

Each real A8 GEMV retains a configurable fraction of the largest-magnitude
activation codes. A tile-K contribution is executed only if that activation is
retained and at least one W4 code in the eight-output tile reaches the weight
threshold. When executed, all eight products are included, matching the
row-masked GEMM model.

| Policy | Measured tile-K drop | PPL | PPL / dense W4A8 | FP32 top-1 | Top-1 agreement with dense W4A8 | KL vs dense W4A8 |
|---|---:|---:|---:|---:|---:|---:|
| W4 >= 1, keep 100% A8 | 14.0% | 2.7103 | 1.0000 | 86.3% | 100.0% | 0.00000 |
| W4 >= 1, keep 90% A8 | 22.3% | 2.7282 | 1.0066 | 86.0% | 94.8% | 0.01870 |
| W4 >= 1, keep 75% A8 | 35.4% | 2.8349 | 1.0460 | 84.2% | 90.3% | 0.06046 |
| W4 >= 2, keep 100% A8 | 39.6% | 3.2458 | 1.1976 | 79.6% | 83.6% | 0.18199 |
| W4 >= 3, keep 100% A8 | 55.8% | 17.9139 | 6.6095 | 43.8% | 44.6% | 1.90220 |
| W4 >= 4, keep 100% A8 | 72.0% | 1283.6221 | 473.6015 | 3.5% | 3.4% | 6.17486 |
| W4 >= 4, keep 50% A8 | 85.8% | 1241.0032 | 457.8769 | 4.9% | 5.0% | 6.11092 |

`W4 >= 1 / keep 100% A8` removes only columns whose eight quantized weights
are all zero. It is exactly equivalent to dense W4A8 and therefore provides a
14.0% zero-skip opportunity without accuracy loss.

Keeping the largest 90% of A8 values is the conservative first approximate
mode: it skips 22.3% of measured tile-K work, adds 0.66% perplexity over dense
W4A8, and lowers FP32 top-1 by 0.3 percentage points. Keeping 75% skips 35.4%
and adds 4.60% perplexity, which is a plausible aggressive mode but needs a
larger held-out set and generation-quality review.

The boundary becomes poor at `W4 >= 2`: a 39.6% drop costs 19.76% perplexity
and 6.7 FP32 top-1 percentage points relative to dense W4A8. Thresholds 3 and
4 are fatal without sparsity-aware retraining.

## Architectural consequence

The first hardware predicate should distinguish two modes:

1. exact zero-skip: execute a tile-K if any of its eight W4 codes is nonzero;
2. approximate magnitude-skip: combine the nonzero tile mask with a
   configurable A8 magnitude threshold.

Do not hard-wire `abs(W4 code) >= 4`. Preserve a bypass/dense mode and make the
weight and activation thresholds programmable. Start validation at
`W4 >= 1 / keep 90% A8`; treat the 75% mode as an optional performance point.
The earlier threshold-4 scheduler figures remain useful only as an intentionally
extreme load-balance stress test.

The evaluator is `budgetcuda.stories_sparsity_accuracy`; the CLI is
`testing/tiny_stories/run_sparsity_accuracy.py`. Generated JSON records both
FP32-relative and incremental dense-W4A8-relative logit metrics.
