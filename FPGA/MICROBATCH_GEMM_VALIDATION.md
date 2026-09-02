# TinyStories micro-computation screening and GEMV-to-GEMM model

## Question

When a decoder produces many tiny candidate computations, does the cost of
deciding whether each candidate is significant erase the benefit of dropping
sparse work? Can surviving GEMVs from different batches be tagged and packed
as rows of a shared GEMM without losing their destinations?

## Model

The model uses the real W4 codes from the 500-step TinyStories QAT checkpoint.
It forms one candidate for every `(batch, operation, output tile, K chunk)`.
The significance predicate is true when an activation marked significant and
an `abs(W4 code) >= 4` weight coincide within the chunk. Activation masks are
deterministic synthetic Bernoulli masks with eight batch factors
`0.05, 0.10, 0.25, 0.50, 1, 1, 1, 1` and a base density of 0.5.

Every survivor retains `(batch_id, operation_id, generation)`. Compatible
survivors that share operation, output tile, and K range fill up to eight rows
of one GEMM. Each row retains its own active-K mask; the multiplier executes
the union of the masks and disables rows that do not need a particular K step.
A multiset comparison between survivor descriptors and packed output tags
detects drops, duplicates, or misrouting.

The front end charges one predicate cycle per predicate-lane group and one
enqueue cycle per compactor-lane group. Two matrix multipliers each charge two
launch cycles plus one cycle per active K step in a bundle. The conservative
bound is `screen + matrix`; the ideal producer-consumer bound is
`max(screen, matrix)`.

This is an architecture-throughput experiment. The synthetic activation masks
and W4 threshold are not an accuracy-approved pruning rule.

## Eight-lane predicate, four-lane survivor compactor

| K chunk | Candidates | Dropped | Survivors | Rows/GEMM | Row fill | Screen | Matrix | Serial | Serial speedup vs dense GEMM | Pipeline speedup vs dense GEMM | Wasted row-steps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 949,248 | 57.5% | 403,432 | 5.10 | 63.8% | 219,514 | 335,196 | 554,710 | 1.925x | 3.186x | 1,807,109 |
| 32 | 474,624 | 48.7% | 243,546 | 5.45 | 68.2% | 120,215 | 300,817 | 421,032 | 2.395x | 3.353x | 2,145,529 |
| 64 | 261,376 | 40.5% | 155,465 | 5.60 | 70.0% | 71,539 | 283,936 | 355,475 | 2.762x | 3.458x | 2,393,730 |

All three configurations preserve every survivor tag with zero mismatches.
The ten-bit composite tag occupies two bytes per survivor: 788 KiB at K=16,
476 KiB at K=32, and 304 KiB at K=64.

K=16 makes the most precise decisions and wastes the fewest matrix row-steps,
but screening 949,248 candidates remains costly. K=64 has the best modeled
throughput, although coarse chunks cause 32.5% more wasted row-steps than
K=16. K=32 is the balanced first RTL point: it is 18.4% slower than K=64 by
the serial bound, but wastes 10.4% fewer row-steps and uses 39.5% fewer tag
bytes than K=16.

## Width sensitivity

A single-lane survivor enqueue is a first-order bottleneck. With eight
predicate lanes, changing enqueue width from one to four changes the serial
bound as follows:

| K chunk | One-lane enqueue | Four-lane enqueue | Reduction |
|---:|---:|---:|---:|
| 16 | 857,284 | 554,710 | 35.3% |
| 32 | 603,691 | 421,032 | 30.3% |
| 64 | 472,073 | 355,475 | 24.7% |

At K=32, two predicate lanes plus four enqueue lanes already balance the
front end and multiplier stages (`298,199` versus `300,817` cycles). Wider
predicate hardware only lowers the non-overlapped serial bound; it does not
improve the ideal pipelined bound once the matrix stage dominates.

## Initial architecture

1. Generate one candidate descriptor per K chunk and evaluate significance in
   a vector predicate stage.
2. Use a four-lane stable compactor so survivor tag writes do not serialize.
3. Store the tag beside the survivor descriptor; do not perform a separate tag
   lookup before execution.
4. Match survivors by `(operation, output tile, K range)` and fill an
   eight-entry GEMM row buffer.
5. Launch a partial bundle on full, timeout, dependency fence, or queue drain.
6. Carry a per-row active-K mask through the multiplier and route every result
   using the row's immutable composite tag.
7. Start RTL with K=32 and at least two predicate lanes. Keep K=64 as the
   throughput configuration and K=16 as the finer-grained accuracy experiment.

## Still missing

- Replace synthetic activation masks with traces from TinyStories inference.
- Measure the predictor itself and validate its pruning decision against
  logits/perplexity.
- Model finite row-buffer capacity, timeout underfill, backpressure, and
  dependency fences in an event-driven pipeline.
- Include packed sparse-index reads and DDR/cache behavior.
- Synthesize the predicate, compactor, tag RAM, and row-mask datapath, then feed
  measured frequency and latency back into the model.

The implementation is in `budgetcuda.microbatch_gemm_model`; the command-line
driver is `testing/tiny_stories/run_microbatch_gemm_model.py`.
