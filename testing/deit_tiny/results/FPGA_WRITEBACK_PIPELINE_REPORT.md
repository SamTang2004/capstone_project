# DeiT-Tiny FPGA pipelined-writeback report

This report compares the same fused six-class W4A8 DeiT-Tiny graph before and
after replacing serialized GEMM requantization/writeback with a pipelined
single-DSP requantizer, an 8x8 INT8 tile buffer, and registered burst writes.
Both captures are from the physical PYNQ-Z2 at 100 MHz with the complete model,
descriptor, and workspaces resident in the 64 MiB DDR arena.

## Result

The optimized image preserves the reference logits `[2, -2, 2, 0, 0, -5]`.
GEMM writeback falls from 29,417,342 to 16,607,624 cycles, a 43.55% reduction
and a saving of 128.10 ms. The complete PL window falls from 217,613,870 to
204,966,023 cycles, a 5.81% reduction from 2.176 to 2.050 seconds. The result is
still 4.10x above the 500 ms target.

The implementation uses 14,252 LUTs (26.79%) and 162 of 220 DSPs (73.64%). It
passes 100 MHz timing signoff with WNS +0.001 ns and WHS +0.016 ns. All 25 RTL
simulations, 62 firmware tests, and 84 host tests pass (one host test skipped).

| Module | Layers | Before cycles | After cycles | After time | Change |
|---|---:|---:|---:|---:|---:|
| GEMM, including fused writeback | 218 | 73,407,830 | 60,131,602 | 601.32 ms | -18.09% |
| Activation-B packing | 120 | 46,123,120 | 46,283,293 | 462.83 ms | +0.35% |
| Standalone requantization | 98 | 35,332,873 | 35,302,213 | 353.02 ms | -0.09% |
| Softmax | 36 | 30,174,739 | 30,161,377 | 301.61 ms | -0.04% |
| LayerNorm | 25 | 19,253,971 | 19,250,176 | 192.50 ms | -0.02% |
| GELU | 12 | 6,920,543 | 6,908,241 | 69.08 ms | -0.18% |
| Residual add | 24 | 4,833,076 | 4,824,861 | 48.25 ms | -0.17% |
| Token embedding | 1 | 1,082,358 | 1,615,390 | 16.15 ms | +49.25% |
| Token concatenation | 12 | 485,360 | 488,870 | 4.89 ms | +0.72% |

The token-embedding increase is a single-launch DDR arbitration variation. The
unmodified large modules remain within normal run-to-run variation. Stall
counters overlap module windows and must not be added to the times above.

## GEMM phases

| GEMM phase | Before cycles | After cycles | After time | Change |
|---|---:|---:|---:|---:|
| Load A | 6,352,465 | 6,342,432 | 63.42 ms | -0.16% |
| Load B | 17,244,944 | 17,242,366 | 172.42 ms | -0.01% |
| Systolic compute | 10,599,617 | 10,599,617 | 106.00 ms | 0.00% |
| Writeback/requantize | 29,417,342 | 16,607,624 | 166.08 ms | -43.55% |

This isolates the speedup to the intended phase. The full-model saving closely
matches the 12,809,718 writeback cycles removed, showing that the tile-buffered
writer did not merely move work into another measured module.

## Remaining bottlenecks

GEMM remains the largest module at 29.34% of the PL window, followed by
activation packing at 22.58%, standalone requantization at 17.22%, Softmax at
14.72%, and LayerNorm at 9.39%. Within GEMM, loading B costs 172.42 ms and the
remaining write phase costs 166.08 ms. Across the full graph, memory stalls
account for 99,869,593 overlapping cycles (48.72% of the PL window).

The next optimization should target the separate activation-packing and
requantization passes, ideally by producing the packed B layout directly from
preceding operators and extending fused INT8 output to the remaining eligible
GEMMs. Softmax/LayerNorm traffic is the next independent target. Adding another
arithmetic core without reducing DDR traffic will have limited effect.

Raw captures are `fpga_w4a8_fused_report.json` and
`fpga_w4a8_pipelined_writeback_report.json` in this directory.
