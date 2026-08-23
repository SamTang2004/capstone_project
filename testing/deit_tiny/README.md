# DeiT-Tiny fruit-quality experiment

This experiment compares DeiT-Tiny W8A8, W4A8 mixed precision, W4A4 integer,
and scaled E2M1 FP4 quantization for six image classes:

1. `apple_fresh`
2. `apple_rotten`
3. `banana_fresh`
4. `banana_rotten`
5. `orange_fresh`
6. `orange_rotten`

## Dataset

The source is the **Fresh and Rotten Fruits Dataset for Machine-Based
Evaluation of Fruit Quality**, DOI `10.17632/bdd69gyhv8.1`, licensed CC BY 4.0.
Only the publisher's 3,200-image original archive is used. The supplied
augmented archive is excluded to prevent transformed copies from leaking
between train and test sets.

The six selected folders contain 1,200 files. Dataset validation found and
excluded seven exact duplicate fresh-orange files, leaving 1,193 unique images:

- training: 835
- validation: 180
- held-out test: 178

Adjacent filename indices are kept in groups of five before the deterministic
70/15/15 assignment. The tracked manifest and audit summary are in `splits/`.
Downloaded archives, cached images, and model checkpoints are ignored by Git.

## Quantization contract

All quantized models start from the same best float fine-tuned checkpoint.

- INT8 is W8A8; mixed precision is W4A8; INT4 is W4A4.
- FP4 uses the E2M1 codebook `{0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3,
  +/-4, +/-6}` for both weights and matrix activations.
- Matrix weights are symmetric and per-output-channel.
- Matrix inputs and outputs are signed symmetric per-tensor activations.
- Attention probabilities are unsigned per-tensor values at the same bit width.
- LayerNorm, GELU, softmax, residual additions, and simulated accumulations
  remain floating point.
- All variants use the same data, initialization, augmentations, seed,
  optimizer family, and 12 QAT epochs.

The FP4 run uses the same scale granularity as INT4 to isolate the element
format. It is not NVFP4: NVFP4 adds hierarchical 16-element block scaling and
FP8 scale metadata.

The table in `RESULTS.md` remains a fake-quantized QAT accuracy comparison. A
separate shared-scale W8A8 hardware manifest can now be exported and executed
on the PYNQ-Z2; its integer scaling contract is intentionally more restrictive
than the per-output-channel QAT experiment.

## Reproduce

Install the pinned experiment dependency and place the official DeiT-Tiny
checkpoint at the path expected by the training script. Then run:

```powershell
python testing/deit_tiny/prepare_dataset.py
python -m pytest testing/deit_tiny/test_quantization.py -q
python testing/deit_tiny/train_deit_tiny.py --device cuda
python testing/deit_tiny/train_mixed_precision.py --device cuda
python testing/deit_tiny/train_fp4.py --device cuda
python testing/deit_tiny/export_budgetcuda.py
```

To independently reload and score a saved model:

```powershell
python testing/deit_tiny/evaluate_checkpoint.py `
  testing/deit_tiny/artifacts/int4/best_fake_quant.pt --device cuda
```

To run a complete W8A8 image through a BudgetCUDA board server:

```powershell
python testing/deit_tiny/run_budgetcuda_deit.py path/to/image.jpg `
  --manifest testing/deit_tiny/artifacts/budgetcuda_w8a8/manifest.json `
  --address 192.168.2.99 --repeat 2
```

Add `--cycle-report testing/deit_tiny/results/fpga_cycle_report.json` to run
with precise 64-bit PL counters and save operation, stage, and per-layer timing.

The runner validates the manifest and uploads one fused full-model graph into a
64 MiB DDR3 arena. Weights, class/position embeddings, activation workspaces,
and the descriptor all remain allocated. `--repeat` therefore performs exactly
one input write, one graph command, and one logits read per image. The physical
W8 graph occupies 44,162,214 bytes including its 98,712-byte descriptor. With
two 8x8 GEMM cores, the measured graph loads once in 49.91 seconds and inference
completes in 4.71 seconds with unchanged logits. The exact PL window is 4.517
seconds, 46.15% below the optimized one-core 8.388-second result and 10.03x
faster than the 45.31-second streamed baseline. It remains above the
less-than-500-ms deployment target.

An experimental shared LayerNorm/Softmax reciprocal engine reduced the exact
PL window to 3.157 seconds, including LayerNorm at 192.55 ms and Softmax at
301.47 ms. Its standalone arithmetic is bit-exact to the legacy units at the
model's 192/197-element dimensions. The rebuilt full fabric nevertheless
shifted final logits while preserving the predicted class, so the board was
rolled back to the known-good dual-core image pending full-fabric isolation.
See `results/FPGA_CYCLE_REPORT.md` for the measured comparison.

See `RESULTS.md` for the measured comparison and limitations.
