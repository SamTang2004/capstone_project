# Wide finder measured results

Generated from completed XSim and isolated synthesis runs. All widths retain eight DSP multipliers.

Standalone finder: 4,662 accepted frames / 36,968 cycles, with 18,424 output-stall cycles and 4,659 simultaneous frame replacements.
Real-DSP compute comparison: **104 result vectors / 762 declared dot results / 188,182 cycles**, all passing.
The scan-8 and dense anchors match the previous experiment on all main comparison fixtures.

## Resident wide-word supply

Command-to-result cycle counts (lower is better).

| Fixture | Useful products | Dense-8 | Finder-8 | Finder-16 | Finder-32 | Finder-64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_w8 | 2048 | 259 | 261 | 261 | 261 | 261 |
| half_w8 | 1024 | 259 | 261 | 260 | 260 | 260 |
| ten_percent_w8 | 800 | 1003 | 1005 | 663 | 583 | 571 |
| point_one_x_99 | 8 | 1003 | 1005 | 509 | 261 | 133 |
| zero_jobs | 0 | 67 | 69 | 37 | 21 | 13 |
| short_jobs | 40 | 13 | 15 | 15 | 15 | 15 |

## Supply limited to eight operand pairs per cycle

Command-to-result cycle counts (lower is better).

| Fixture | Useful products | Dense-8 | Finder-8 | Finder-16 | Finder-32 | Finder-64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_w8 | 2048 | 259 | 261 | 262 | 264 | 268 |
| half_w8 | 1024 | 259 | 261 | 262 | 264 | 268 |
| ten_percent_w8 | 800 | 1003 | 1005 | 1007 | 1015 | 1008 |
| point_one_x_99 | 8 | 1003 | 1005 | 1005 | 1005 | 1005 |
| zero_jobs | 0 | 67 | 69 | 69 | 69 | 69 |
| short_jobs | 40 | 13 | 15 | 16 | 16 | 16 |

## Useful utilization with resident input

Useful products divided by eight times the complete command-to-result interval; this includes scanning and drain time.

| Fixture | Finder-8 | Finder-16 | Finder-32 | Finder-64 |
| --- | ---: | ---: | ---: | ---: |
| half_w8 | 49.042% | 49.231% | 49.231% | 49.231% |
| ten_percent_w8 | 9.950% | 15.083% | 17.153% | 17.513% |
| point_one_x_99 | 0.100% | 0.196% | 0.383% | 0.752% |

The 64-position finder uses **7.56x fewer cycles** than the 8-position finder on the extreme resident fixture. With the eight-pair supply cap, that advantage disappears.
Random 50% density rarely leaves a completely empty eight-position bank. One-bank-per-cycle arbitration limits the gain even when input words are wide.

## Synthesis cost

| Scan positions | LUTs | Registers | DSPs | BRAM | Setup WNS (ns) | Hold WHS (ns) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 3967 | 1764 | 8 | 0 | 1.145 | -0.041 |
| 16 | 4461 | 1897 | 8 | 0 | 0.954 | -0.041 |
| 32 | 5372 | 2163 | 8 | 0 | 0.741 | -0.041 |
| 64 | 7156 | 2703 | 8 | 0 | 0.256 | -0.041 |

Post-synthesis estimates at a 10 ns clock target with 0.300 ns uncertainty, xc7z020clg400-1. No place/route, external IO timing, physical clock location, memory banking, DDR or board measurements are included. Positive setup slack is not timing closure.

The resident-width comparison increases available input-word width along with finder width. The capped-supply comparison charges acquisition from the first frame onward. The W4 stall fixture is correctness-only because its injected source bubbles are per frame, not per original position.

[Architecture and reproduction](WIDE_FINDER.md) · [Raw metrics](WIDE_FINDER_PERFORMANCE.csv)
