# Dual-lane GEMM writeback and rowwise burst report

The PYNQ-Z2 was measured at 100 MHz with the timing-clean W4A8 image. The
implementation retains 0.300 ns additional setup uncertainty and passes with
WNS +0.022 ns and WHS +0.029 ns. It uses 166 DSPs (75.45%) and 42.5 BRAM tiles
(30.36%). All 27 RTL regressions, W8A8/W4A8 transformer graph smokes, and both
HeatViT selector modes pass.

## Architecture

- The fused GEMM requantizer now captures two INT32 accumulator results per
  cycle through two registered multiplier lanes. A full 8x8 tile reaches
  `capture_done` within 38 cycles and retains the existing packed burst writer.
- The shared LayerNorm/Softmax DDR wrapper buffers eight output rows in BRAM,
  transposes the row stream into tiled memory order, and writes contiguous
  32-bit words. An 8x4 tiled output therefore uses eight words and one burst
  terminator instead of 32 masked, individually terminated writes.
- Softmax arithmetic is unchanged: subtract the row maximum, evaluate a
  17-entry Q15 `exp(-distance/4)` LUT for distances 0 through 16, clamp larger
  distances to zero, compute one Q32 reciprocal per row with the iterative
  divider, then use reciprocal multiplication and an exact remainder
  correction to produce rounded Q0.7 probabilities.

## Physical cycle results

| Graph | Previous PL cycles | New PL cycles | New PL time | Change |
|---|---:|---:|---:|---:|
| DeiT-Tiny W4A8 | 105,483,961 | 85,007,541 | 850.075 ms | -19.41% |
| Balanced HeatViT W4A8 | 72,248,572 | 59,194,965 | 591.950 ms | -18.07% |

Both runs retain their expected fresh-apple classification and exact logits:
baseline `[2, -2, 2, 0, 0, -5]`, HeatViT `[1, -3, 1, 1, 0, -6]`.

HeatViT is now 9,194,965 cycles (91.950 ms) above the 500 ms target. A further
15.53% reduction from this result is required.

## Module cycles

| Module | Baseline before | Baseline after | HeatViT before | HeatViT after |
|---|---:|---:|---:|---:|
| GEMM | 40,489,857 | 37,481,286 | 27,978,269 | 25,920,284 |
| Softmax | 28,771,743 | 16,681,597 | 17,012,798 | 9,909,709 |
| LayerNorm | 19,225,336 | 13,864,247 | 13,891,855 | 10,024,066 |
| GELU | 7,472,395 | 7,468,960 | 5,465,128 | 5,463,531 |
| Residual add | 5,032,299 | 5,023,868 | 3,721,376 | 3,717,867 |
| Token concat | 3,425,593 | 3,421,335 | 2,526,134 | 2,524,431 |
| Token embedding | 1,066,738 | 1,066,248 | 1,082,838 | 1,067,638 |
| Token prune | - | - | 570,174 | 567,439 |

Softmax falls by 42.02% in the baseline and 41.75% with HeatViT. LayerNorm
falls by 27.89% and 27.84%, respectively, because it shares the same row-tile
burst path.

## GEMM and stall counters

| Counter | Baseline before | Baseline after | HeatViT before | HeatViT after |
|---|---:|---:|---:|---:|
| GEMM Load A | 6,596,149 | 6,593,520 | 4,687,585 | 4,685,067 |
| GEMM Load B | 11,895,707 | 11,959,565 | 7,502,746 | 7,626,865 |
| GEMM compute | 10,599,617 | 10,599,617 | 7,585,688 | 7,585,688 |
| GEMM write | 13,836,852 | 10,913,461 | 9,409,941 | 7,370,519 |
| Total output stall | 19,238,002 | 1,250,106 | 12,079,398 | 797,341 |
| Total memory stall | 50,071,435 | 31,690,902 | 33,203,517 | 21,682,379 |

The dual-lane writer reduces GEMM write by 21.13% for the baseline and 21.67%
for HeatViT. Rowwise bursts reduce total output stalls by 93.50% and 93.40%.

Raw reports:

- `fpga_cycle_report_100mhz_write2_rowburst_baseline.json`
- `fpga_cycle_report_100mhz_write2_rowburst_heatvit.json`
