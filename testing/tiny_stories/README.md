# TinyStories host reference

The host path reads the legacy `llama2.c` `stories15M.bin` checkpoint without
copying its FP32 tensors, runs deterministic single-sequence inference with a
KV cache, and exports full vocabulary logits plus per-layer golden tensors.

Export a token-ID trace:

```powershell
python testing/tiny_stories/export_golden.py artifacts/stories15M.bin `
  --tokens 1,1724,338 --new-tokens 8 `
  --output testing/tiny_stories/results/once_upon_a_time
```

Export from text when `tokenizer.bin` is available:

```powershell
python testing/tiny_stories/export_golden.py artifacts/stories15M.bin `
  --tokenizer artifacts/tokenizer.bin --prompt "Once upon a time" `
  --new-tokens 8 --output testing/tiny_stories/results/once_upon_a_time
```

The `.npz` file contains all FP32 arrays. Its adjacent `.json` records model
identity, tensor shapes and hashes, complete token flow, and the highest logits
for quick inspection. The exporter requires the exact 15M model dimensions by
default; `--allow-compatible-model` is intended only for development fixtures.

Validate both the archive hash and every tensor payload before consuming a
golden in firmware or RTL tests:

```powershell
python testing/tiny_stories/validate_golden.py `
  testing/tiny_stories/results/once_upon_a_time_fp32.json
```

## WXAY sensitivity sweep

The fake-quant sweep compares every configuration against the same FP32 logits.
Weights use symmetric quantization; activations are dynamically quantized at
operator boundaries; attention probabilities use an unsigned range; and
accumulation remains FP32 to model a sufficiently wide INT32 accumulator.
RMSNorm parameters remain FP32. These tests isolate quantization sensitivity;
they do not yet reproduce a fixed-point firmware datapath.

Run a focused set of configurations on the official validation corpus:

```powershell
python testing/tiny_stories/run_quantization_sweep.py `
  testing/tiny_stories/artifacts/stories15M.bin `
  testing/tiny_stories/artifacts/tokenizer.bin `
  --corpus testing/tiny_stories/artifacts/TinyStories-valid.txt `
  --stories 8 --tokens-per-story 32 `
  --configs W8A8,W8A6,W8A4,W6A8,W6A6,W6A4,W4A8,W4A6,W4A4,W3A8 `
  --weight-granularity per_output `
  --output testing/tiny_stories/results/wxay_official_valid_per_output
```

The initial 248-token validation slice found the following boundary:

- `W8A8` is effectively lossless, and `W6A8`/`W6A6` remain viable.
- `W4A8` is degraded but survives with per-output-channel weight scales; it
  retained the tested greedy sample but increased perplexity by about 28%.
- `W4A6` is severe and should not be a deployment target without calibration
  or quantization-aware training.
- A4 is fatal for every tested weight width, and W3 is fatal even with A8.
- Per-tensor `W4A8` is fatal (about 1447x FP32 perplexity), demonstrating that
  scale granularity is part of the architecture contract rather than metadata.

Grades are project sensitivity labels: `S` is near-lossless, `A` mild, `B`
material, `C` severe, and `F` fatal. Fatal means at least one of: perplexity
ratio >= 2, FP32 top-1 agreement < 0.50, mean logit cosine < 0.90, or a
non-finite result. A larger held-out evaluation and an exact integer reference
remain required before freezing the board format.

## W4A8 QAT

`train_w4a8_qat.py` fine-tunes the original checkpoint with per-output-channel
W4 weights, per-vector dynamic A8 activations, unsigned A8 attention
probabilities, and straight-through estimators. The default loss combines
next-token cross entropy with KL distillation from a frozen original FP32
teacher. The exported artifact is a normal llama2.c float-shadow checkpoint;
the NumPy fake-quant reference remains the authoritative evaluator.

```powershell
python testing/tiny_stories/train_w4a8_qat.py `
  testing/tiny_stories/artifacts/stories15M.bin `
  testing/tiny_stories/artifacts/tokenizer.bin `
  testing/tiny_stories/artifacts/TinyStories-train-prefix.txt `
  --output-dir testing/tiny_stories/artifacts/w4a8_qat_500step `
  --steps 500 --batch-size 4 --sequence-length 64 `
  --learning-rate 1e-5 --distillation-weight 0.5 --device cuda
```

On a disjoint 32-story/2,016-token slice of the official validation set:

| Model | PPL | FP32 ratio | FP32 top-1 | Logit cosine | KL |
|---|---:|---:|---:|---:|---:|
| Original FP32 | 2.3612 | 1.0000 | 100.0% | 1.0000 | 0.0000 |
| W4A8 PTQ | 3.0746 | 1.3021 | 81.0% | 0.95885 | 0.27094 |
| W4A8 QAT, 500 steps | 2.7103 | 1.1479 | 86.3% | 0.96727 | 0.15145 |

This short run recovered about 51% of the PTQ perplexity gap while retaining
the same packed W4 weight bandwidth. The QAT shadow-FP32 checkpoint itself has
PPL 2.4746, so some original float quality moved into quantization robustness.
The result is promising but remains grade B; more diverse training, periodic
held-out checkpoint selection, learned/clipped activation scales, and an exact
integer deployment simulation are the next experiments.

Export the QAT checkpoint through the actual W4A8 fake-quant reference:

```powershell
python testing/tiny_stories/export_quantized_golden.py `
  testing/tiny_stories/artifacts/w4a8_qat_500step/stories15M_w4a8_qat_float.bin `
  --source-checkpoint testing/tiny_stories/artifacts/stories15M.bin `
  --tokenizer testing/tiny_stories/artifacts/tokenizer.bin `
  --prompt "Once upon a time" --weight-bits 4 --activation-bits 8 `
  --weight-granularity per_output --new-tokens 8 `
  --output testing/tiny_stories/results/once_upon_a_time_w4a8_qat
```

The resulting `budgetcuda-stories-wxay-golden` archive contains 12 decode
steps and 1,260 tensors: full 32K logits, operator-boundary dequantized
fake-quant activations, per-layer tensors, and K/V cache prefixes. Its JSON
records both the QAT checkpoint and original source-checkpoint SHA-256 values,
the complete quantization policy, token flow, shapes, and per-tensor hashes.
The current full-model golden proves the host fake-quant contract. The module
set below adds packed INT4 codes and exact INT32 accumulators; fixed-point scale
encodings and bit-exact hardware requantizer parameters still require the final
deployment scale contract.

## Per-module W4A8 integer goldens

`stories_module_goldens.py` provides independent Python simulators for
embedding gather, W4A8 linear/GEMM, RMSNorm, RoPE, K/V update, causal
attention, Softmax/probability quantization, residual add, SiLU, SwiGLU, final
normalization, and vocabulary projection. W4A8 GEMM archives include logical
INT4/INT8 operands, the exact accelerator tiled byte layouts, INT32
accumulators, per-output scales, pre-requant values, and output INT8 codes.

Generate all modules for every layer at the final prompt position:

```powershell
python testing/tiny_stories/export_module_goldens.py `
  testing/tiny_stories/artifacts/w4a8_qat_500step/stories15M_w4a8_qat_float.bin `
  --tokenizer testing/tiny_stories/artifacts/tokenizer.bin `
  --prompt "Once upon a time" --capture-position 4 `
  --output-directory `
    testing/tiny_stories/results/modules_w4a8_qat_once_upon_a_time_pos004

python testing/tiny_stories/validate_module_goldens.py `
  testing/tiny_stories/results/modules_w4a8_qat_once_upon_a_time_pos004/manifest.json
```

This produces 99 module archives containing 1,089 independently hashed
tensors. Earlier prompt positions are executed first to populate the real K/V
prefix. Each archive is written to a temporary file and atomically renamed;
the manifest is written last, so an interrupted export cannot present a
partial set as complete.

The integer simulator uses explicit INT32 GEMM accumulation, whereas the
earlier functional fake-quant reference multiplies dequantized FP32 operands.
At position 4 their top-1 token agrees (`29892`), but their maximum absolute
logit difference is `0.373634` after dynamic requantization through six
layers. They are deliberately recorded as separate contracts; the integer
module golden is the appropriate target for firmware and RTL.

## Sparse scheduler overhead model

The work-stealing validation model derives tile costs from the actual QAT W4
codes, preserves all 43 GEMM dependency barriers, and compares static
round-robin, a serialized central queue, and distributed work stealing.

```powershell
python testing/tiny_stories/run_work_stealing_model.py `
  testing/tiny_stories/artifacts/w4a8_qat_500step/stories15M_w4a8_qat_float.bin `
  --workers 2 --significant-code 4 --chunks whole,64,32,16 `
  --output testing/tiny_stories/results/work_stealing_overhead_w4_code4
```

The first result favors complete-K output-tile tasks: 6,232 tasks and 97.4 KiB
of 16-byte descriptors. With a one-cycle local pop and eight-cycle steal,
work stealing is neutral on the naturally balanced order (`1.002x`) and
recovers `1.137x` over static scheduling in the pathological distribution.
See `FPGA/WORK_STEALING_VALIDATION.md` for assumptions and the proposed local
queue architecture.
