# BudgetCUDA research report: overview

## Purpose and intended audience

This report evaluates BudgetCUDA as a small-model Transformer accelerator and
research prototype. It is being prepared as supporting material for an
application to a research-based master's programme; the expected readers may
therefore include researchers who will look for a precise research question,
reproducible evidence, explicit limitations, and a defensible distinction
between implemented work and proposed work.

The present system targets a PYNQ-Z2 FPGA. Its deployed numerical contract is
primarily W8A8: signed eight-bit activations and weights with wider
accumulation. A packed W4A8 path is also implemented, but the current
per-tensor scaling contract trades away substantially more accuracy than the
research quantizer. The design is not merely a matrix-multiplication demo: it
contains a host graph builder and runtime, firmware execution layer, DDR-backed
Transformer operators, an 8-by-8 DSP GEMM array, an integrated bitstream, and
physical end-to-end DeiT-Tiny measurements.

## Provisional research question

> How effectively can a resource-constrained FPGA execute useful sub-64 MiB
> quantized Transformer models through a CUDA-like resident graph abstraction,
> and which architectural or numerical constraints dominate accuracy,
> latency, and model compatibility?

This wording is deliberately broader than fruit classification while remaining
anchored to what the repository can test. It makes model compatibility,
quantization, memory residency, and utilization measurable research concerns
rather than presenting the project only as an implementation exercise.

## Current system and strongest evidence

The implemented accelerator provides tiled W8A8 and packed-W4A8 GEMM,
LayerNorm, GELU, Softmax, residual addition, activation packing, token
concatenation, token construction, and token pruning. A full 12-block,
three-head DeiT-Tiny graph has been executed on the PYNQ-Z2 with model constants
resident in DDR. The latest correctness-qualified implementation reported in
the experiment timeline is a dual-core, W4A8, balanced-HeatViT design at 100
MHz. It uses 165 of 220 DSPs, 26,695 of 53,200 LUTs, and 42 of 140 BRAM tiles,
with +0.006 ns setup and +0.033 ns hold slack. Its average resident device graph
took 416.63 ms and the host-observed path took 440.27 ms while preserving the
expected logits. This is the appropriate performance baseline; the earlier
8.93-second, single-core W8 graph is an implementation milestone, not the
current system.

The corresponding correctness-qualified counter capture records 1.8045 billion
operations in 41,686,928 cycles: 4.335 GOPS, or 16.9% of the dual-array 25.6
GOPS arithmetic peak. The accelerator was busy for 86.8% of the graph window,
while memory stalls occupied 41.4% of that window. These metrics are not
contradictory: a pipeline may be active while waiting for DDR or performing
operators that cannot reach the GEMM arithmetic peak. The evidence therefore
supports a memory/scheduling-utilization bottleneck, but not the earlier claim
that the current pipeline achieves only 2.2% of peak.

### Newest continuous-resident experiment

An even newer performance experiment loads the model once and submits 20
serialized resident graphs back-to-back. It measured 327.37 ms mean device
time, 352.62 ms mean host time, and 2.835 images/s. The isolated diagnostic run
recorded 84.0% accelerator busy time, 5.542 GOPS, and 21.65% arithmetic-peak
utilization. Across the continuous host wall time, estimated accelerator busy
time was 77.68%, effective throughput was 5.115 GOPS, and arithmetic-peak
utilization was 19.98%. Cross-image hardware overlap was disabled, so these
figures already show that residency and immediate resubmission materially
improve efficiency without inter-image pipelining.

This newest capture is currently **performance evidence, not a
correctness-qualified inference result**. A later transfer-correctness probe
produced saturated logits `[127, 127, 127, 127, 127, 127]`, rather than the
expected model output. The 326--353 ms timing remains informative because the
same graph shapes and operation schedule executed, but it must not replace the
416.63 ms result as the headline correct-inference measurement until the data
transfer or model-loading fault is resolved and exact logits are recovered.

The detailed newest counters sharpen the bottleneck diagnosis. GEMM occupies
58.9% of the graph window and is busy for 95.1% of its own windows, yet 32.0%
of GEMM-window cycles are memory stalls. For the entire graph, memory stalls
account for 27.1%. Static W4 weights already achieve a 99.4% cache hit rate;
the remaining priorities are dynamic-B traffic, output writeback and overlap,
plus firmware-executed token concatenation and embedding. In short, the fully
resident pipeline is much healthier than the original overview implied, while
data movement remains a measured optimization boundary.

The best local accuracy evidence is more nuanced than a single preferred bit
width. On the 178-image held-out split, the research W8A8 and W4A8 QAT models
reached 100.00% and 99.44% respectively. However, the scalar-requantization
deployment experiments reported 98.88% for W8A8 and only 21.91% for W4A8.
Consequently, W4 storage is an important research direction, but it is not yet
the defensible default for the existing hardware scale contract.

These are promising engineering results, not yet general accelerator claims.
The dataset is small, one physical image is not a board-level accuracy study,
the measured inference path is far below peak throughput, and the ASIC results
are exploratory OpenROAD estimates rather than sign-off measurements.

## Experiment 1: can the architecture express sub-64 MiB language models?

The first experiment is a static compatibility probe, not a language-quality
benchmark. The script
[`experiments/micro_model_fit.py`](experiments/micro_model_fit.py) combines
analytic storage and operation counts with actual construction of one
attention-plus-MLP block through the repository's graph builder. It tests five
synthetic shapes chosen to expose the design boundaries rather than to claim
compatibility with a particular pretrained checkpoint. Generated measurements
are stored in
[`experiments/micro_model_fit_results.json`](experiments/micro_model_fit_results.json).

The current hard or software-enforced bounds used by the probe are:

- no more than three attention heads in the existing graph builder;
- a maximum GEMM reduction dimension of 1024 in the RTL scheduler;
- a maximum hardware Softmax vector length of 256;
- dense, bidirectional attention semantics; and
- INT8 activations, with either INT8 or packed INT4 matrix weights.

### Preliminary interpretation

A BERT-tiny-shaped encoder block (sequence 128, hidden width 128, two heads,
MLP width 512) maps directly onto the existing attention and MLP graph. A
12-layer byte-level encoder (sequence 256, hidden width 192, three heads, MLP
width 768) also stays within the current dimensional contracts. These are
credible near-term language experiments because their core block uses operators
that are already implemented.

The measured static estimates are small: the BERT-tiny-shaped configuration is
3.92 MiB at W4 and 4.11 MiB at W8 when its proposed token embedding is stored as
INT8; the 12-layer byte encoder is 2.67 MiB at W4 and 5.21 MiB at W8. A more
aggressive 12-layer decoder shape with width 768, three 256-wide heads, MLP
width 1024, and an 8,192-token vocabulary remains under the requested bound at
approximately 28.8 MiB W4 or 51.3 MiB W8 (assuming tied input/output
embeddings). This demonstrates storage and dense-block fit, not runnable
autoregressive semantics.

A small decoder-only Transformer can fit well below 64 MiB by parameter count
and its dense GEMMs can be described, but it cannot yet be called an LLM
deployment on this accelerator. Correct autoregressive inference additionally
requires a causal attention mask, token-embedding lookup, positional encoding
(or an equivalent such as RoPE), persistent KV-cache updates, vocabulary-logit
handling, and token selection. None is part of the present end-to-end graph
contract. Recomputing the entire prefix with unmasked attention would be both
semantically wrong and inefficient.

The result is therefore asymmetric:

- **Small encoder language models:** architecturally close and suitable for the
  next experiment, especially byte/character models or compact classifiers.
- **Micro decoder language models:** storage-compatible and GEMM-compatible,
  but not execution-compatible without several new stateful and causal
  operators.
- **Existing DeiT/HeatViT direction:** still the most mature application and
  the correct baseline against which any language experiment should be judged.

### Concrete pretrained language-model candidates

The phrase *micro-LLM* needs qualification. Models that fit this device can be
genuine pretrained autoregressive Transformers, but they are not compressed
general-purpose assistants. The defensible research claim is **micro language
model inference for a constrained domain**, followed by measured scaling—not
that a sub-64 MiB model retains the capabilities associated with billion-
parameter LLMs.

The strongest first target is Karpathy's `stories15M` checkpoint from
[`llama2.c`](https://github.com/karpathy/llama2.c). It is a Llama-style causal
model trained on TinyStories with hidden width 288, six layers, six attention
heads, and context length 256. The published FP32 binary is 60,816,028 bytes;
the weight payload therefore has a lower-bound footprint of approximately
14.5 MiB at W8 or 7.25 MiB at W4, before quantization scales and runtime state.
Its dimensions are close to this accelerator: the context already matches the
256-element Softmax limit and its dense reductions fit the 1024 scheduler
limit. Six-head scheduling, RMSNorm, RoPE, SiLU/gated MLP, causal masking,
embedding lookup, KV-cache mutation, and sampling still need implementation.

The [`TinyStories` study](https://arxiv.org/abs/2305.07759) is an academically
useful basis for this experiment because it explicitly studies whether models
below ten million non-embedding parameters can produce coherent, grammatical
stories in a deliberately restricted domain. This makes generated-text
quality, perplexity, exact CPU/FPGA agreement, prefill latency, decode latency,
and energy per token meaningful evaluation targets without pretending the
model is a general assistant.

Other candidates expose useful tradeoffs:

- [`roneneldan/TinyStories-1M`](https://huggingface.co/roneneldan/TinyStories-1M)
  is a released GPT-Neo causal LM. Its
  [configuration](https://huggingface.co/roneneldan/TinyStories-1M/blob/main/config.json)
  uses width 64, eight layers, sixteen heads, a 50,257-token vocabulary, and
  alternating local/global attention. Its FP32 safetensors file is about 48.6
  MB, so its full stored parameter set is only roughly 12 MiB at W8. It fits
  memory easily but its sixteen very narrow heads and GPT-Neo attention modes
  make it a less direct architectural match than `stories15M`.
- [`Pythia-70M`](https://huggingface.co/EleutherAI/pythia-70m) is a much more
  general pretrained causal LM. Its official
  [configuration](https://huggingface.co/EleutherAI/pythia-70m/blob/main/config.json)
  uses six layers, width 512, eight heads, MLP width 2048, a 50,304-token
  vocabulary, RoPE, and untied embeddings. An ideal all-W4 payload can fit in
  the arena in principle, but the current contract would keep embeddings at
  INT8 and leave almost no runtime headroom. Moreover, the 2048-wide MLP
  exceeds the current reduction limit, the eight heads require generalized
  scheduling, and the present W4 numerical contract is not accurate enough.
  It is a second-stage target.
- [`SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) is too
  large for the present arena: an ideal four-bit payload alone is approximately
  64.4 MiB before scales, KV cache, activations, or descriptors. Its
  [configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M/blob/main/config.json)
  also requires 30 layers, nine query heads, grouped-query attention, RMSNorm,
  RoPE, SiLU, and a 1536-wide MLP.

Accordingly, `stories15M` is the recommended proof model. It is small enough
that W8 can be used first, avoiding the unresolved W4 accuracy problem, while
remaining a real pretrained decoder capable of producing inspectable text.

## How the micro-model experiment links to full LLM inference

The link is architectural and procedural, not a claim that model quality is
scale-invariant. `stories15M` uses the same decoder-only autoregressive loop as
larger Llama-family models: token lookup, repeated pre-normalized attention and
gated-MLP blocks, a final normalization and vocabulary projection, followed by
token selection. The key model operations—RMSNorm, RoPE, causal attention,
SwiGLU/SiLU, residual connections, KV caching and dense projections—do not
change meaning when layer count and tensor dimensions increase. Published
architecture comparisons at the 7--8B scale likewise identify RMSNorm, RoPE,
full or grouped-query attention and SwiGLU as the defining Llama-like block
components ([OLMo architecture comparison](https://arxiv.org/pdf/2402.00838)).

What does change with scale is weight capacity, external-memory bandwidth,
context/KV-cache capacity, attainable parallelism and quantization sensitivity.
Those differences must be modelled and measured rather than hidden. A PYNQ-Z2
execution of `stories15M` cannot establish the accuracy or absolute throughput
of a 7B model. It can establish that the accelerator implements the same
inference state machine and can calibrate how its tiled kernels and DDR path
scale with width, layers, context and precision.

### Required end-to-end inference contract

The experiment represents actual LLM inference only if it implements both
standard phases:

1. **Prefill:** process all prompt tokens with a causal mask and populate the
   per-layer KV cache while producing the first next-token distribution.
2. **Decode:** process one new token at a time, append its keys and values,
   attend to the cached prefix, project to vocabulary logits and select the next
   token without recomputing the prefix.

This distinction is established in current LLM systems research: prefill
exposes parallel prompt computation, whereas small-batch decode is commonly
memory-bandwidth-bound and depends on persistent KV state
([KVQuant, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/028fcbcf85435d39a40c4d61b42c99a4-Abstract-Conference.html)).
It also aligns directly with FPGA LLM work: FlightLLM identifies computation,
memory overhead, bandwidth use and mapping/compilation as the central FPGA
inference problems
([FlightLLM](https://arxiv.org/abs/2401.03868)). Running unmasked attention or
recomputing the complete prefix at every generated token would therefore be a
text-generation demonstration, but not representative LLM inference.

### Evidence ladder and permitted claims

The report should build its claim in four levels:

1. **Operator equivalence:** every decoder operator agrees with a CPU integer
   reference over random, boundary and model-derived tensors.
2. **Algorithmic equivalence:** CPU and FPGA execute the same quantized prefill,
   KV-cache update and decode recurrence, with matching logits and token
   sequences under deterministic sampling.
3. **Workload equivalence:** measurements report time to first token, decode
   tokens/s, end-to-end latency, DDR bytes, memory stalls and energy/token over
   prompt lengths 16, 64, 128 and 256. These are the same workload quantities
   used to reason about larger serving systems; MLPerf, for example, includes
   Llama 2 70B server and offline inference scenarios
   ([MLPerf documentation](https://docs.mlcommons.org/inference/benchmarks/language/llama2-70b/)).
4. **Scaling model:** a counter-calibrated compute-and-bandwidth model predicts
   latency and capacity as width, MLP ratio, head count, layers, vocabulary and
   context increase. Synthetic shape sweeps should test the model up to the
   hardware limits. Predictions beyond 64 MiB must be labelled projections for
   a larger-memory FPGA, not PYNQ-Z2 results.

Passing these levels would support the claim that BudgetCUDA is a
**configurable quantized Transformer inference accelerator demonstrated on both
vision encoding and autoregressive language decoding**. It would not support a
claim that the PYNQ-Z2 runs a contemporary general-purpose LLM, nor that
TinyStories quality predicts chat, reasoning or retrieval performance.

### Connection to actual applications

The completed causal path maps directly to constrained text completion,
streaming story generation, command or template completion, predictive text,
and scoring/ranking candidate strings by likelihood. Chat formatting,
retrieval-augmented generation and tool invocation use the same inference loop,
but useful behavior in those applications primarily requires a larger or more
capable trained model, longer context and an external orchestration layer. They
are therefore architectural use cases and scaling targets, not capabilities to
claim from the TinyStories checkpoint itself. Training and fine-tuning remain
outside the proposed accelerator scope.

For the *general-purpose* label to be defensible, the implementation must be
configuration-driven rather than `stories15M`-specific: head count, mask mode,
normalization, activation, positional transform, MLP width, vocabulary and
context must be descriptor parameters or compiler choices. DeiT and
`stories15M` should pass through the same runtime and GEMM/memory substrate,
with only their declared graphs differing. Supporting these two substantially
different model families is stronger evidence of generality than running
several checkpoints with the same architecture.

## Recommended next experiments

1. Import `stories15M` into a CPU reference harness, record its exact tensor
   inventory and generated-token baseline, and perform W8 post-training or
   quantization-aware evaluation before changing the FPGA graph contract.
2. First resolve the saturated-logit transfer/model-loading fault, then repeat
   the 20-image continuous-resident benchmark with exact-logit assertions. Use
   the now-established detailed counter format for both the encoder and
   DeiT-Tiny so their GEMM, memory-stall, and firmware shares are comparable.
3. Add generic embedding lookup, six-head scheduling, RMSNorm, RoPE,
   causal Softmax, SiLU/gated MLP, and explicit KV-cache updates incrementally.
   Validate each new operator against the `stories15M` CPU reference before
   attempting complete generation.
4. Treat W4A8 as a controlled quantization experiment: compare per-tensor,
   grouped, and per-channel weight scales before claiming a practical 4-bit
   language-model path.

## Scope of the full report

The eventual report should include: motivation and research gap; related work;
system requirements; host, firmware, RTL, and memory architecture; numerical
formats and quantization; experimental methodology; FPGA implementation and
timing/resource results; accuracy and performance results; ASIC exploration;
limitations and threats to validity; and a prioritized research roadmap. Every
headline result should be traceable to a command, generated artifact, test, or
hardware log, and externally sourced claims should be cited from primary
literature or official documentation.

## Repository evidence used in this overview

- `README.md`
- `FPGA/FULL_DEIT_READINESS.md`
- `FPGA/TRANSFORMER_MIGRATION.md`
- `FPGA/hardware/README.md`
- `testing/deit_tiny/RESULTS.md`
- `ASIC/RESULTS.md`
- `FPGA/budgetcuda/deit.py`
- `FPGA/hardware/rtl/transformer_gemm_scheduler.sv`
- `FPGA/hardware/rtl/rowwise_norm_softmax_engine.sv`
- `testing/deit_tiny/results/FPGA_IMPLEMENTATION_TIMELINE.md`
- `testing/deit_tiny/results/fpga_cycle_report_100mhz_fused_shared_correct.json`
- `testing/deit_tiny/results/fpga_cycle_report_current_detailed.json`
- `testing/deit_tiny/results/fpga_continuous_benchmark.json`
- `testing/deit_tiny/results/fpga_transfer_correctness_probe.json`
