# Significance of the tagged sparse scheduler

Research comparison reviewed 2026-09-03. The comparison separates three
claims that must not be conflated: finding accuracy-safe sparsity, converting
sparsity into wall-clock speedup, and dynamically batching irregular work.

## Bottom line

The current TinyStories result is a useful feasibility point, but is not yet a
competitive sparsity result by itself.

- Exact W4 zero-skip removes 14.0% of tile-K work with no numerical change.
  This is an important baseline, not a novel pruning result.
- Keeping the largest 90% of A8 values removes 22.3%, raises perplexity by
  0.66% over dense W4A8, and changes 5.2% of dense-W4A8 top-1 decisions.
  Its ideal compute-only upper bound is just `1/(1-0.223) = 1.287x` before
  screening, compaction, memory, and non-GEMM overhead.
- Keeping 75% removes 35.4% and has a `1.548x` ideal upper bound, but raises
  perplexity by 4.60% and changes 9.7% of dense-W4A8 top-1 decisions.
- The original W4-code threshold 4 is an accuracy-invalid stress workload.

In comparison, published activation-sparsity work commonly targets 40-50% or
more. TEAL reports 40-50% model-wide training-free sparsity and up to 1.53x and
1.8x decoding speedups. CATS reports roughly 50% activation sparsity with
downstream performance often within 1-2%, although its custom kernel improves
token-generation latency by only about 15%. These results also demonstrate why
removed operations alone are not an end-to-end speedup claim.

## Closest prior work

| Work | Sparsity/selection | Training or prediction | Reported quality | Reported realized benefit | Relation to this project |
|---|---|---|---|---|---|
| TEAL (ICLR 2025) | Magnitude-based, 40-50% model-wide activation sparsity | Training-free; layerwise sparsity allocation | Minimal degradation across Llama-2/3 and Mistral, 7B-70B | Up to 1.53x at 40%, 1.8x at 50% | Stronger accuracy-sparsity baseline than our current 22.3% point |
| CATS (2024) | Context-aware thresholding in gated MLPs, about 50% activation sparsity | New nonlinear activation; can run without fine-tuning | Often within 1-2% of base downstream performance | About 15% token-generation latency improvement | Direct evidence that mask/compaction overhead can dominate moderate sparsity |
| DejaVu (2023) | Input-dependent MLP neurons and attention heads; paper observes up to 95% contextual sparsity in MLPs | Lightweight learned predictors with asynchronous lookahead | Reported no loss of model quality/in-context learning | More than 2x vs FasterTransformer for OPT-175B | Closest predictor-overhead precedent; our predicate must be cheaper and hardware-measured |
| ProSparse (2025 revision) | ReLUfication plus progressive regularization; around 88-89% activation sparsity | Continued training and threshold shifting | Comparable to original Swish models on reported benchmarks | Exact PowerInfer path up to about 1.7x in its table | Shows what sparsity-aware retraining can recover; much higher training cost |
| SparseGPT (ICML 2023) | Static weight pruning; at least 50%, up to 60% unstructured | One-shot layer reconstruction, no retraining | Minimal/negligible perplexity increase on large GPT models | Sparsity result rather than a universal hardware speedup | Strong static-pruning accuracy baseline, but not our dynamic non-N:M target |
| Wanda (2023) | Weight magnitude times calibrated activation norm; commonly 50% | Calibration data, no weight update | Competitive with SparseGPT; superior to magnitude-only pruning | 2:4 linear layers about 1.6x; LLaMA-7B end-to-end 1.24x | Demonstrates why raw weight magnitude thresholding is too weak |
| Orca (OSDI 2022) | No pruning; iteration-level scheduling and selective batching | Runtime serving scheduler | Exact | 36.9x throughput vs FasterTransformer at matched latency in its GPT-3 experiment | Cross-request GEMV-to-GEMM batching is established prior art |
| vLLM/PagedAttention (SOSP 2023) | No pruning; memory-aware continuous batching | Runtime request scheduling | Exact | 2-4x serving throughput over compared systems | Tagged batch identity and dynamic admission are standard serving concepts |
| MegaBlocks (MLSys 2023) | Dynamic, imbalanced MoE routing represented as block-sparse operations | Learned MoE router; dropless execution | Avoids token dropping | Up to 40% over Tutel and 2.4x over dense Megatron-LM training | Closest block-sparse/grouped-GEMM precedent for irregular tagged work |

## What may still be distinctive

The individual ingredients are established. A defensible contribution would
have to be their FPGA-specific composition:

1. a low-latency W4A8 tile-significance predicate that is cheaper than the
   micro-work it rejects;
2. stable multi-lane compaction that preserves `(batch, operation, generation)`
   without a separate tag lookup;
3. packing surviving rows from independent GEMVs into a row-masked GEMM;
4. idle-triggered work stealing between local queues when sparse row counts are
   imbalanced;
5. programmable exact and approximate policies on the same datapath.

This is narrower than claiming a new sparsity method or dynamic batching. Its
value would be that a small W4A8 FPGA can retain dense-array efficiency under
fine-grained, input-dependent irregularity without N:M constraints.

## Evidence required for a significant result

The next milestone should use the following minimum bar:

- report real board end-to-end tokens/s, latency, energy/token, LUT/FF/BRAM/DSP,
  frequency, and DDR traffic against the same dense W4A8 design;
- beat dense W4A8 by at least 1.2x at a policy with no more than 1% PPL growth,
  or demonstrate a clearly superior energy/latency Pareto point;
- separately compare exact zero-skip, magnitude screening without stealing,
  tagged packing without stealing, and the full work-stealing design;
- include predicate, compactor, queue, timeout-underfill, and tag-routing cycles;
- replace the synthetic scheduler mask with per-layer real TinyStories traces;
- validate more than 2,016 tokens and preferably a second model, because a 15M
  TinyStories model does not establish scaling to modern LLMs.

At present, the conservative 22.3% point is below the sparsity level usually
needed for a compelling sparse kernel. It is meaningful only if the proposed
FPGA machinery converts it into measurable speed or energy gains at unusually
low overhead. The 35.4% point is closer to the literature's performance region
but needs sparsity-aware QAT to reduce its 4.60% perplexity penalty.

## Primary sources

- TEAL, *Training-Free Activation Sparsity in Large Language Models*:
  https://arxiv.org/abs/2408.14690
- CATS, *Contextually-Aware Thresholding for Sparsity in Large Language Models*:
  https://arxiv.org/abs/2404.08763
- DejaVu, *Contextual Sparsity for Efficient LLMs at Inference Time*:
  https://arxiv.org/abs/2310.17157
- ProSparse, *Introducing and Enhancing Intrinsic Activation Sparsity within
  Large Language Models*: https://arxiv.org/abs/2402.13516
- SparseGPT, *Massive Language Models Can Be Accurately Pruned in One-Shot*:
  https://arxiv.org/abs/2301.00774
- Wanda, *A Simple and Effective Pruning Approach for Large Language Models*:
  https://arxiv.org/abs/2306.11695
- Orca, *A Distributed Serving System for Transformer-Based Generative Models*:
  https://www.usenix.org/conference/osdi22/presentation/yu
- vLLM/PagedAttention, *Efficient Memory Management for Large Language Model
  Serving with PagedAttention*: https://arxiv.org/abs/2309.06180
- MegaBlocks, *Efficient Sparse Training with Mixture-of-Experts*:
  https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html
