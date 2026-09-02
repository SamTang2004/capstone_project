# stories15M port: gap analysis and work list

## Objective

Run Karpathy's pretrained `stories15M` TinyStories checkpoint on the existing
PYNQ-Z2 BudgetCUDA stack without removing or changing the behavior of the
working DeiT/HeatViT path.

The first target is correct W8A8 autoregressive generation at a maximum context
of 256 tokens. W4 is a later optimization, not part of the first bring-up.

## Target model contract

The published `stories15M` checkpoint uses the llama2.c/Llama architecture:

- transformer width: 288;
- feed-forward width: 768;
- layers: 6;
- query heads: 6;
- KV heads: 6;
- head width: 48;
- vocabulary: 32,000;
- maximum context: 256;
- tied token-embedding and output-classifier weights;
- pre-norm RMSNorm, RoPE, causal self-attention, and a SiLU-gated MLP.

All dense reductions fit the current `MAX_K=1024` scheduler. The 256-token
context fits the current rowwise/Softmax length. W8 GEMM can also traverse the
32,000-column vocabulary projection because the scheduler's `MAX_N=768` bound
only constrains its present W4 cache modes.

## Arena fit

The FP32 checkpoint is about 60.8 MB, corresponding to roughly 15.2 million
stored scalar parameters. Its lower-bound W8 weight payload is about 14.5 MiB.

Additional persistent W8 runtime state is small:

- INT8 K cache: `6 * 256 * 288 = 442,368` bytes;
- INT8 V cache: another 442,368 bytes;
- INT32 vocabulary logits: 128,000 bytes;
- current-token activations and attention scratch: well below 1 MiB with
  planned buffer reuse;
- scale tables, descriptors, tokenizer metadata, and alignment: to be measured
  by the exporter rather than estimated away.

The tied 32,000-by-288 embedding/classifier tensor needs special care. The
current GEMM expects a tiled `288 x 32,000` matrix, whereas a conventional
embedding lookup expects row-major `32,000 x 288`. Duplicating it would cost an
extra 9,216,000 bytes (8.79 MiB) but would still fit. The preferred solution is
one tiled copy plus a column-gather embedding operation, preserving weight tying
and arena headroom.

**Static verdict:** W8 fits comfortably. This must still be confirmed by an
exported package and the real allocator, including all padding and transient
buffers.

W4 does not currently execute this model even though its raw bytes fit. The W4
cache launch constraints reject important `288 x 768`, `768 x 288`, and
`288 x 32,000` matrices. General W4 streaming/cache support and language-model
accuracy work are separate later milestones.

## Existing components that can be reused

- 64 MiB contiguous DDR arena, resident model packaging, graph dispatch, and
  host read/write protocol;
- W8 INT8-by-INT8 GEMM with INT32 accumulation and fused requantized output;
- dimensions needed by all attention and MLP projections;
- residual addition and INT8 activation packing/layout conversion;
- rowwise Softmax arithmetic as a basis for length-aware attention Softmax;
- existing LayerNorm reciprocal/square-root machinery as a basis for RMSNorm;
- graph diagnostics and per-module cycle counters;
- mock backend, firmware unit-test harness, RTL simulations, and PYNQ smoke-test
  structure.

The existing `TOKEN_EMBEDDING` opcode is not a vocabulary lookup. It combines
DeiT patch, class, and position embeddings using a scale LUT, so its opcode and
behavior must remain unchanged.

## Missing correctness components

These are required before the board can claim real autoregressive inference.

### Checkpoint import and integer reference

- [x] Add a llama2.c checkpoint reader that validates the header and exact
      `stories15M` dimensions before converting any tensor.
- [x] Import the official tokenizer and keep tokenization/detokenization on the
      host for the first version.
- [x] Build a floating-point CPU reference runner from the imported tensors.
- [ ] Freeze the W8 weight, activation, attention-score, cache, and residual
      scale contracts. The host sensitivity sweep now establishes a starting
      policy, but it does not yet model the exact fixed-point datapath.
- [ ] Build an integer CPU reference with exactly the rounding, saturation,
      RMSNorm, RoPE, Softmax, and SiLU approximations intended for firmware.
- [x] Export deterministic golden traces for at least one prompt: token IDs,
      per-layer tensors, K/V cache slices, logits, and selected tokens.
- [ ] Emit a complete arena ledger from the packaged model and fail export if
      the 64 MiB bound is exceeded.

The completed host components live in `budgetcuda.stories`. The first real
checkpoint trace uses the prompt `Once upon a time`, includes eight greedy
tokens, and is generated locally under `testing/tiny_stories/results`. The
checkpoint reader also exposes a source-tensor ledger, but that is not yet the
final quantized 64 MiB arena ledger.

### Additive graph ABI

- [ ] Reserve new opcodes; do not repurpose `LAYERNORM`, `GELU`, `SOFTMAX`, or
      the DeiT-specific `TOKEN_EMBEDDING` opcode.
- [ ] Add a vocabulary-embedding gather that can read a token column directly
      from the tied tiled classifier tensor, or explicitly choose the simpler
      duplicated row-major table for bring-up.
- [ ] Add `RMSNORM` with no mean subtraction and no beta term.
- [ ] Add `ROPE` for Q and K using the current decode position.
- [ ] Add length-aware/causal Softmax that excludes cache entries after the
      current position.
- [ ] Add persistent K/V cache write and prefix-read semantics for each layer.
- [ ] Add `SILU_MUL` implementing `SiLU(w1(x)) * w3(x)` before `w2`.
- [ ] Generalize attention construction from three to six heads. If the layer
      descriptor remains limited to three inputs, concatenate heads in a small
      tree rather than changing the existing token-concat ABI.
- [ ] Add final RMSNorm and the tied 32,000-way vocabulary projection.

### Stateful decoder runtime

- [ ] Add an LLM model package/runtime separate from `DeiTRuntime`.
- [ ] Represent token ID and current position as small per-invocation input
      buffers while weights and K/V cache remain resident.
- [ ] Add explicit cache reset and reject positions outside `0..255`.
- [ ] Build one-token decode recurrence first. Use the same recurrence to ingest
      prompt tokens, producing a correct but initially slow prefill path.
- [ ] Ensure mutable K/V buffers survive repeated graph executions and are not
      accidentally activation-aliased or overwritten by model reload logic.
- [ ] Keep sampling on the board-side ARM or host initially. Avoid transferring
      128 KiB of INT32 logits over TCP for every generated token if only one
      sampled token is needed.
- [ ] Implement deterministic greedy decoding first; add temperature/top-k/
      top-p only after exact greedy agreement.
- [ ] Define BOS/EOS handling and stop conditions exactly as the checkpoint's
      tokenizer expects.

### Firmware first, RTL second

- [ ] Implement reference-correct firmware fallbacks for vocabulary gather,
      RMSNorm, RoPE, length-aware Softmax, KV update/read, and SiLU-multiply.
- [ ] Dispatch all dense projections through the existing PL W8 GEMM path.
- [ ] Generate text with the hybrid firmware/PL implementation before adding
      new RTL accelerators.
- [ ] Profile the working decoder and migrate only measured bottlenecks into
      RTL. Likely candidates are RMSNorm/RoPE fusion, decode attention over the
      KV cache, SiLU-multiply, and local vocabulary reduction/sampling.

## Regression and acceptance gates

### Preserve current functionality

- [ ] Keep all existing opcode numbers and descriptor meanings unchanged.
- [ ] Keep the current DeiT/HeatViT exporter and runtime independent of the new
      LLM package.
- [ ] Run the full host and firmware test suites after every ABI milestone.
- [ ] Run transformer RTL simulations after each top-level control change.
- [ ] Before accepting a new bitstream, reproduce the existing known image
      logits and HeatViT/DeiT board smoke results.

### Language-model correctness

- [ ] Check every imported tensor's shape, orientation, checksum, and packed
      integer samples.
- [ ] Match the integer CPU reference exactly for each new primitive.
- [ ] Match layer-by-layer tensors for a one-token decode step in the mock and
      firmware backends.
- [ ] Match K/V cache contents and greedy tokens over a multi-token prompt and
      at least 32 generated tokens.
- [ ] Measure floating-point versus W8 validation loss/perplexity on a fixed
      TinyStories subset and set an explicit acceptable degradation threshold.
- [ ] Demonstrate correct generation at positions 0, 1, 254, and 255, including
      cache reset and EOS termination.

### Board fit and performance

- [ ] Load the complete package into the real 64 MiB arena and record weights,
      persistent state, transient peak, descriptors, padding, and free bytes.
- [ ] Record model-load time, prompt ingestion time, time to first token,
      steady-state tokens/s, per-token latency, and energy/token if measurable.
- [ ] Use counters to separate vocabulary projection, attention/KV traffic,
      MLP GEMMs, normalization, and sampling.
- [ ] Treat optimized full-sequence prefill as a later performance milestone;
      do not block the first correct decoder on it.

## Ordered milestones

1. **M0 — Reproducible fit:** checkpoint parser, tensor inventory, W8 packer,
   arena ledger, tokenizer, and floating/integer CPU references.
2. **M1 — Mock decoder:** additive opcodes and a complete six-layer decode graph
   with exact golden-trace agreement in the mock backend.
3. **M2 — Hybrid board generation:** firmware fallbacks plus existing PL GEMMs;
   greedy text generation with persistent KV cache and unchanged DeiT results.
4. **M3 — Measurement:** board memory proof, perplexity/quality evaluation, and
   a per-component latency profile.
5. **M4 — Acceleration:** move measured decoder bottlenecks into RTL and add an
   optimized causal prefill path.
6. **M5 — Optional W4:** generalize W4 matrix handling, establish a language
   accuracy contract, and compare W4 against the working W8 baseline.

## Initial WXAY sensitivity boundary

The host fake-quant harness in `budgetcuda.stories_quantization` measures
teacher-forced perplexity, FP32 top-1 agreement, logit cosine/NRMSE/KL, first
greedy divergence, and generated text. On 248 prediction tokens from the
official TinyStories validation set, per-output-channel weight scales gave:

- W8A8: 1.009x FP32 perplexity (near-lossless);
- W6A8: 1.029x and W6A6: 1.114x (viable candidates);
- W4A8: 1.282x (survives, but already severe enough to require calibration or
  quantization-aware training before adoption);
- W4A6: 1.599x (severe/borderline);
- W8A4/W6A4/W4A4 and W3A8: fatal under the project threshold.

With one scale per complete layer tensor, W4A8 rose to 1447x FP32 perplexity.
The initial architecture should therefore target W8A8. W6A6 is the next
compression experiment. W4 must use a finer weight-scale granularity and must
not be treated as a drop-in extension of the present coarse-scale path.

A first 500-step W4A8 QAT run used 512 stories from a disjoint official
training prefix plus FP32-teacher distillation. On 2,016 held-out validation
tokens, it reduced PPL from the PTQ result of 3.0746 to 2.7103 against an FP32
PPL of 2.3612. Top-1 agreement improved from 81.0% to 86.3%, KL fell from
0.27094 to 0.15145, and about 51% of the PTQ perplexity gap was recovered.
This preserves W4 bandwidth but is still grade B, so it supports continued W4
research rather than replacing the W8A8 bring-up target.

The 500-step checkpoint now also has a validated W4A8 functional golden for
`Once upon a time`: 12 decode steps, 1,260 traced tensors, full vocabulary
logits, and all K/V cache prefixes. This is the dequantized fake-quant golden;
the module set below adds packed INT4 weights and INT32 accumulators, while
fixed-point scale encodings must follow the final deployment scale contract.

The first integer-observable module set is also available for prompt position
4. It contains 99 independently hashed module archives and 1,089 tensors
covering all six layers. W4A8 linear archives include accelerator-packed
operands, INT32 accumulators, scales, and requantized outputs. The explicit
integer pipeline and functional dequantized-matmul reference agree on top-1
token 29892, although their maximum logit difference is 0.373634 after six
layers; firmware/RTL should target the integer module contract.

A scheduler validation model now uses the real QAT W4 codes and all 43 GEMM
dependency barriers. Its first recommendation is whole-K, 8-output tile tasks
with one-cycle local dequeues and steal-on-empty behavior. This needs 6,232
tasks/97.4 KiB of descriptors per token, remains within 3.2% of the modeled
sparse ideal, and improves the deliberately pathological two-worker mapping by
13.7%. K-chunk subdivision is deferred because descriptor and partial-result
overhead dominate. See `WORK_STEALING_VALIDATION.md`.

The batch-tag extension now carries `(batch_id, operation_id, generation)`
through actual steal/completion events. Two batches require an eight-bit tag,
adding 12.2 KiB across 12,464 tasks. At one-cycle lookup, 5%/100% batch skew is
1.785x faster than static with zero routing errors; the modeled tag-lookup
crossover is 39/40 cycles per task. Equal loads trigger no steals and lose
about 1.1%, so the first hardware policy should steal only when locally idle.

The micro-computation extension now charges significance screening before
discard, compacts surviving tagged GEMVs, and packs compatible work as rows of
an eight-row GEMM. The first hardware point is K=32 with a four-lane survivor
compactor and at least two predicate lanes: its modeled screen and matrix
stages are balanced at 298,199/300,817 cycles. K=64 remains the throughput
option, while K=16 is reserved for finer pruning experiments. All modeled
survivors retain their tags; accuracy still requires real activation traces.

## Definition of the first successful port

The port is complete at the initial W8 milestone only when the PYNQ-Z2 loads
the real pretrained checkpoint into the 64 MiB arena, consumes a tokenized
prompt, maintains a correct per-layer KV cache, and generates at least 32 greedy
tokens matching the integer reference, while the existing DeiT/HeatViT tests
and known board logits remain unchanged.
