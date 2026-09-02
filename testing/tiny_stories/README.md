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
The current golden proves the host fake-quant contract. Packed INT4 codes,
fixed-point scale encodings, and exact accumulator/requantizer integers belong
in the later integer-reference golden once that deployment contract is frozen.
