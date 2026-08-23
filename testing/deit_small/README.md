# DeiT-Small fruit-quality experiment

This experiment tests whether the extra capacity of canonical DeiT-Small can
recover accuracy under the same aggressive quantization contract used for
DeiT-Tiny. It reuses the exact six classes, 1,193 unique images, grouped
70/15/15 split, preprocessing, seed, optimizer family, and training schedule
documented in `../deit_tiny/README.md`.

## Model and quantization

- Model: `deit_small_patch16_224.fb_in1k`, 21,667,974 parameters with a
  six-class head (3.92 times DeiT-Tiny).
- Initialization: official DeiT-Small ImageNet-1k checkpoint
  `deit_small_patch16_224-cd65a155.pth`.
- Float fine-tuning: 20 epochs; QAT: 12 epochs per format.
- INT8 is W8A8 and INT4 is W4A4.
- FP4 is scaled E2M1 fake quantization, not NVFP4 or MXFP4.
- Weights are symmetric per-output-channel. Matrix activations are signed
  per-tensor; attention probabilities are unsigned per-tensor.
- LayerNorm, GELU, softmax, residual additions, and accumulation remain float.

This is an accuracy experiment with fake-quantized QAT. It is not a packed
hardware export or a PYNQ latency measurement.

## Reproduce

The generalized scripts remain in `../deit_tiny/` so Tiny and Small use one
implementation:

```powershell
python testing/deit_tiny/train_deit_tiny.py `
  --device cuda `
  --model-name deit_small_patch16_224.fb_in1k `
  --pretrained-checkpoint testing/deit_tiny/data/pretrained/deit_small_patch16_224-cd65a155.pth `
  --output-dir testing/deit_small/artifacts

python testing/deit_tiny/train_fp4.py `
  --device cuda `
  --float-checkpoint testing/deit_small/artifacts/float/best.pt `
  --output-dir testing/deit_small/artifacts
```

See `RESULTS.md` for the measured Tiny-versus-Small comparison.
