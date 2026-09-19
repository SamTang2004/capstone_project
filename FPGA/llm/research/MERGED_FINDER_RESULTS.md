# Cross-bank merger measured results

Same eight-DSP arithmetic core, fixtures, input widths and source-supply models as the bank-dispatch baseline.

Standalone merged finder: **4,729 frames**, 16,577 chunks, 33,150 cycles; 16,562 stall cycles and 4,724 simultaneous frame replacements. Checks include every byte/threshold/position, every survivor count 0–64, full intermediate beats, paired payloads, indices, metadata, and reset after a partially drained frame.
Real-DSP comparison: **104 result vectors / 762 dot results**, passing in 186,956 scoreboard cycles. Includes W4 thresholds, source/result stalls, INT32 wrap, reset and back-to-back waves. Default bank-dispatch regression and both scan-8/dense controls match the saved metrics exactly.

## Resident wide words

Command-to-result cycles, shown as **old bank dispatch → merged**.

| Fixture | Scan 8 | Scan 16 | Scan 32 | Scan 64 |
| --- | ---: | ---: | ---: | ---: |
| dense_w8 | 261 → 261 | 261 → 261 | 261 → 261 | 261 → 261 |
| half_w8 | 261 → 261 | 260 → 187 | 260 → 161 | 260 → 145 |
| ten_percent_w8 | 1005 → 1005 | 663 → 509 | 583 → 261 | 571 → 155 |
| point_one_x_99 | 1005 → 1005 | 509 → 509 | 261 → 261 | 133 → 133 |
| zero_jobs | 69 → 69 | 37 → 37 | 21 → 21 | 13 → 13 |
| short_jobs | 15 → 15 | 15 → 15 | 15 → 15 | 15 → 15 |

## Supply capped at eight operand pairs per cycle

Command-to-result cycles, shown as **old bank dispatch → merged**.

| Fixture | Scan 8 | Scan 16 | Scan 32 | Scan 64 |
| --- | ---: | ---: | ---: | ---: |
| dense_w8 | 261 → 261 | 262 → 262 | 264 → 264 | 268 → 268 |
| half_w8 | 261 → 261 | 262 → 262 | 264 → 263 | 268 → 265 |
| ten_percent_w8 | 1005 → 1005 | 1007 → 1005 | 1015 → 1005 | 1008 → 1005 |
| point_one_x_99 | 1005 → 1005 | 1005 → 1005 | 1005 → 1005 | 1005 → 1005 |
| zero_jobs | 69 → 69 | 69 → 69 | 69 → 69 | 69 → 69 |
| short_jobs | 15 → 15 | 16 → 16 | 16 → 16 | 16 → 16 |

## Complete-interval utilization at scan 64

Useful products / (eight lanes × command-to-result cycles), with resident input.

| Fixture | Old utilization | Merged utilization | Cycle speedup |
| --- | ---: | ---: | ---: |
| half_w8 | 49.231% | 88.276% | 1.79x |
| ten_percent_w8 | 17.513% | 64.516% | 3.68x |
| point_one_x_99 | 0.752% | 0.752% | 1.00x |

Merging removes partial-bank drain cycles. It does not remove original-position scanning or frame-tail overhead. The extreme sparse fixture is unchanged because it rarely has multiple nonempty banks in a frame. The resident gains depend on wide local operand availability; the capped experiment still pays acquisition from the first frame. W4 source-stall fixtures are correctness-only: bubbles are injected per frame and are not comparable across widths.

## Isolated synthesis

| Width | Design | LUTs | Registers | DSPs | BRAM | Setup WNS ns | Hold WHS ns |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | Bank dispatch | 3967 | 1764 | 8 | 0 | 1.145 | -0.041 |
| 8 | Merged | 4038 | 1765 | 8 | 0 | 0.651 | -0.041 |
| 16 | Bank dispatch | 4461 | 1897 | 8 | 0 | 0.954 | -0.041 |
| 16 | Merged | 4971 | 1899 | 8 | 0 | 0.952 | -0.041 |
| 32 | Bank dispatch | 5372 | 2163 | 8 | 0 | 0.741 | -0.041 |
| 32 | Merged | 7896 | 2170 | 8 | 0 | -2.028 | -0.041 |
| 64 | Bank dispatch | 7156 | 2703 | 8 | 0 | 0.256 | -0.041 |
| 64 | Merged | 13313 | 2703 | 8 | 0 | -6.724 | -0.041 |

**The merged design misses the 10 ns setup target at widths 32, 64.** The combinational prototype demonstrates the cycle benefit but is not ready to claim 100 MHz operation at those widths. A registered or staged merger should be evaluated next, including its added latency and backpressure.

Vivado 2024.2, xc7z020clg400-1, 10 ns clock with 0.300 ns uncertainty before synthesis. These are unplaced estimates, not measured Fmax or timing closure. No IO delays, physical clock placement, SRAM banking, DMA, DDR loading or board run are included. A cycle reduction is not yet a hardware-time or energy improvement.

[Architecture and reproduction](MERGED_FINDER.md) · [Raw metrics](MERGED_FINDER_PERFORMANCE.csv)
