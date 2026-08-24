# DeiT-Tiny quantization results

## Main result

| Model | Validation accuracy | Test accuracy | Test macro F1 | Estimated packed parameter size |
|---|---:|---:|---:|---:|
| Float fine-tuned | 100.00% | 99.44% (177/178) | 99.44% | 21.1 MiB FP32 |
| W8A8 QAT | 100.00% | **100.00% (178/178)** | **100.00%** | 5.47 MiB |
| W4A8 QAT | 99.44% | **99.44% (177/178)** | **99.44%** | 2.86 MiB |
| W4A4 QAT | 61.11% | **65.17% (116/178)** | **64.15%** | 2.86 MiB |
| Scaled E2M1 FP4 QAT | 66.11% | **73.03% (130/178)** | **73.22%** | 2.86 MiB |

Under this aggressive contract, INT4 lost **34.83 percentage points** of test
accuracy relative to INT8. INT8 showed no accuracy loss on this test split.

W4A8 recovered **34.27 of those 34.83 points**, matching the float model's
99.44% test accuracy while retaining four-bit matrix-weight storage. Its only
error was one rotten apple classified as a rotten orange. This isolates the
catastrophic W4A4 loss primarily to activation and attention-probability
precision, not to four-bit weights alone. The 2.86 MiB estimate covers packed
parameters; eight-bit activation buffers and traffic remain a hardware cost.

The FPGA-compatible W4A8 variant uses one per-tensor weight scale rather than
per-output-channel scales. It scores **97.75% (174/178)** test accuracy with
97.76% macro F1, so the deployable scale contract costs 1.69 percentage points
relative to the research W4A8 result while retaining useful accuracy.

FP4 recovered **7.87 percentage points** over INT4 at the same four-bit element
width, but remained **26.97 points below INT8**. Its nonuniform E2M1 spacing is
therefore helpful but insufficient under uniform four-bit activation and
attention quantization. Packed-size estimates exclude scale metadata.

## W4A4 class behavior

| Class | Correct / test | Accuracy |
|---|---:|---:|
| Fresh apple | 12 / 30 | 40.00% |
| Rotten apple | 23 / 30 | 76.67% |
| Fresh banana | 22 / 30 | 73.33% |
| Rotten banana | 29 / 30 | 96.67% |
| Fresh orange | 19 / 28 | 67.86% |
| Rotten orange | 11 / 30 | 36.67% |

W4A4 most strongly compromised fresh-apple and rotten-orange recognition.
Fourteen of the 30 rotten oranges were predicted as rotten apples, indicating
that low-bit quantization damaged fruit identity as well as freshness cues.

## FP4 class behavior

| Class | Correct / test | Accuracy |
|---|---:|---:|
| Fresh apple | 20 / 30 | 66.67% |
| Rotten apple | 21 / 30 | 70.00% |
| Fresh banana | 27 / 30 | 90.00% |
| Rotten banana | 28 / 30 | 93.33% |
| Fresh orange | 18 / 28 | 64.29% |
| Rotten orange | 16 / 30 | 53.33% |

FP4 materially recovered fresh-apple, fresh-banana, and rotten-orange
recognition. Orange quality remains the weakest part of the model.

## Interpretation

The result supports the concern observed with the CNN: a uniform end-to-end
four-bit matrix path is too aggressive for this DeiT-Tiny configuration. W4A8,
however, is accuracy-competitive with both float and W8A8 while halving packed
matrix-weight storage relative to W8A8. It is now the strongest deployment
candidate. Further layer-sensitive experiments are optional optimizations,
rather than prerequisites for recovering accuracy.

## Limitations

- The held-out test set contains only 178 images from one published dataset.
- The dataset supplies no capture-session or fruit-instance identifiers.
  Grouping adjacent filename indices in sets of five is a leakage-reduction
  heuristic, not a guarantee of instance-independent evaluation.
- Perfect INT8 accuracy may partly reflect dataset-specific backgrounds or
  acquisition conditions and must not be presented as field accuracy.
- The accuracy table measures fake-quantized QAT. The shared-scale W8A8 export
  now has packed integer weights and one successful full-image PYNQ-Z2 smoke
  run, but the complete 178-image board accuracy study has not been performed.
- Per-output-channel weight scales improve accuracy but are not supported by
  the repository's current per-tensor BudgetCUDA deployment contract.
- The physical per-tensor W4A8 run occupies 41,433,510 resident bytes, including
  2,870,904 bytes of packed/model weight allocations. A 72 KiB BRAM tile cache
  reduces GEMM B-load from 951.5 to 148.5 ms and the exact PL window from 3.576
  to 2.729 seconds with bit-exact logits. GEMM is now 857.1 ms, requantization
  809.0 ms, and activation packing 436.8 ms; see
  `results/FPGA_W4A8_CACHE_REPORT.md`.
- The streamed-weight baseline took 45.31 seconds. Keeping all 5,599,352 bytes
  of model constants, including class/position embeddings, resident in DDR3
  produced a final 28.48-second physical run with unchanged logits. The
  remaining transient workspaces, descriptors, and 28 graph commands mean this
  is still not evidence for the <500-ms latency goal.
- A 64 MiB arena now holds the complete 44,162,214-byte working set, including
  one 666-layer descriptor. Two 8x8 GEMM cores process separate M tiles while
  sharing each B tile. The exact PL window is 4.517 seconds and the complete
  host-observed inference is 4.708 seconds with unchanged logits: one input
  write, one graph command, and one output read. This is a 46.15% reduction
  from the optimized one-core 8.388-second result. GEMM falls from 5.200 to
  1.887 seconds, but requantization, LayerNorm, Softmax, and activation packing
  still total about 2.50 seconds. Remaining latency is therefore device
  execution efficiency, not intermediate host transfers.
- A dedicated shared reciprocal engine reduced LayerNorm from 723.81 to 192.76
  ms and Softmax from 527.78 to 301.67 ms. Prefix-graph captures found that the
  initial logit shift came from the DDR wrapper reading gamma/beta one column
  late after each first LayerNorm output. A next-column BRAM prefetch fixes the
  alignment, and a distinct-per-column integration test prevents regression.
  Corrected PL windows of 3.197 and 3.143 seconds are 29.22% to 30.42% below
  the dual-core baseline and restore exact logits `[2, -2, 1, 2, -1, -7]`.
  The latest module report attributes 40.43% of the window to GEMM and 39.66%
  to requantization plus activation packing, making a fused GEMM-to-tiled-INT8
  output path the next architectural target.
- The FP4 experiment is scaled E2M1 with matched INT4 scale granularity. It is
  not NVIDIA NVFP4 or OCP MXFP4 and does not include their block-scale metadata.
- Fusing GEMM bias and requantization into packed INT8 writeback contracts the
  W4A8 graph from 666 to 546 layers and reduces the exact PL window from 2.729
  to 2.176 seconds (20.26%) with unchanged logits `[2, -2, 2, 0, 0, -5]`.
  The registered DSP implementation passes 100 MHz timing at WNS +0.001 ns and
  WHS +0.051 ns. See `results/FPGA_OPERATOR_FUSION_REPORT.md`.
- Pipelining fused requantization and buffering each 8x8 INT8 output tile cuts
  GEMM writeback from 294.17 to 166.08 ms (43.55%). The exact PL window falls
  from 2.176 to 2.050 seconds (5.81%) with unchanged logits. The physical image
  uses 162 DSPs and passes 100 MHz timing at WNS +0.001 ns and WHS +0.016 ns.
  See `results/FPGA_WRITEBACK_PIPELINE_REPORT.md`.
- Wide 64-bit W4 cache fills, cache reuse across M passes, fused-writeback
  overlap, and the streamlined shared rowwise engine reduce the timing-clean
  exact PL window from 1.1618 to 1.0957 seconds (5.69%). Layer 0 is byte-exact
  and full logits remain `[2, -2, 2, 0, 0, -5]`. A read/write arbitration race
  that had silently dropped the final word of two-word output rows is fixed and
  covered by regression. GEMM is now 450.83 ms, Softmax 287.49 ms, and
  LayerNorm 192.16 ms. See
  `results/FPGA_BANDWIDTH_ROWWISE_OPT_REPORT.md`.

## Reproducibility

- Model: `deit_tiny_patch16_224.fb_in1k`, 5,525,574 parameters after replacing
  the classifier with six outputs
- Pretraining: official DeiT-Tiny ImageNet-1k checkpoint
  `deit_tiny_patch16_224-a1311bcf.pth`
- Float fine-tuning: 20 epochs
- Quantization-aware training: 12 epochs per precision
- Seed: 42
- Batch size: 32
- Optimizer: AdamW
- Float learning rate: 3e-5
- QAT learning rate: 1e-5
- Weight decay: 0.05
- Hardware used for training: NVIDIA GeForce RTX 3090 Ti
# Full operator fusion

The timing-qualified fused production build runs in 116,181,263 PL cycles
(1.16181263 s at 100 MHz), a 43.32% reduction from the pipelined-writeback
baseline, while preserving exact logits. The complete per-module breakdown is
documented in [results/FPGA_FULL_OPERATOR_FUSION_REPORT.md](results/FPGA_FULL_OPERATOR_FUSION_REPORT.md).
