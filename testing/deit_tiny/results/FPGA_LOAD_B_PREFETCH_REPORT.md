# W4 Load-B prefetch hardware report

The diagnostics-v5 W4A8 image was measured on the PYNQ-Z2 at 100 MHz. The
implemented design passes post-route timing with 0.300 ns additional setup
uncertainty (WNS +0.027 ns, WHS +0.009 ns). W8A8 and W4A8 transformer graph
smokes passed, as did both HeatViT selector modes.

## Full-model latency

| Graph | Previous mean PL cycles | New repeat PL cycles | New mean | Change |
|---|---:|---:|---:|---:|
| DeiT-Tiny W4A8 | 110,543,499 | 105,526,573; 105,483,961 | 105,505,267 | -4.56% |
| Balanced HeatViT W4A8 | 75,520,663 | 72,273,070; 72,248,572 | 72,260,821 | -4.32% |

The second-run PL times are 1.054840 s for the baseline and 0.722486 s for
HeatViT. Both runs classify the fresh-apple test image correctly. The baseline
logits remain `[2, -2, 2, 0, 0, -5]`; balanced HeatViT produces
`[1, -3, 1, 1, 0, -6]`, matching its previously measured output.

HeatViT is still 22,260,821 cycles (222.608 ms) over the 50,000,000-cycle
target. Reaching 500 ms requires another 30.81% reduction from the new mean.

## Last-run module cycles

| Module | Baseline | Balanced HeatViT |
|---|---:|---:|
| GEMM | 40,489,857 | 27,978,269 |
| Softmax | 28,771,743 | 17,012,798 |
| LayerNorm | 19,225,336 | 13,891,855 |
| GELU | 7,472,395 | 5,465,128 |
| Residual add | 5,032,299 | 3,721,376 |
| Token concat | 3,425,593 | 2,526,134 |
| Token embedding | 1,066,738 | 1,082,838 |
| Token prune | - | 570,174 |

## GEMM phase counters

| Phase | Previous baseline | New baseline | Previous HeatViT | New HeatViT |
|---|---:|---:|---:|---:|
| Load A | 6,584,342 | 6,596,149 | 4,677,761 | 4,687,585 |
| Load B | 16,723,931 | 11,895,707 | 11,185,012 | 7,502,746 |
| Compute | 10,599,617 | 10,599,617 | 7,585,688 | 7,585,688 |
| Write | 13,699,388 | 13,836,852 | 9,287,762 | 9,409,941 |

Load B falls by 28.87% for the baseline and 32.92% for HeatViT. Diagnostics-v5
attributes the new baseline Load-B total to 11,424,895 weight cycles and
470,812 bias cycles; HeatViT uses 7,119,401 weight cycles and 383,345 bias
cycles. The baseline records 33,863 W4 cache-hit tiles and 146 misses, while
HeatViT records 24,791 hits and the same 146 compulsory misses.

The full GEMM window falls by 10.90% for the baseline and 10.70% for HeatViT.
Softmax is now the largest non-GEMM target, while GEMM writeback remains the
largest individual GEMM phase.

Raw diagnostics are in:

- `fpga_cycle_report_100mhz_loadb_prefetch_baseline.json`
- `fpga_cycle_report_100mhz_loadb_prefetch_heatvit.json`
