# TileLoader Control Pipeline Comparison

## Implemented Structure

- The final burst beat registers `final_row_r` and `final_col_r` before the
  `ADVANCE` state updates coordinates and selects the next operation.
- `BURST_CALC` registers the five-bit minimum burst length before
  `BURST_PREP` drives the run counters, request state, and error handling.
- Interior AXI burst beats remain contiguous; the added cycles occur only at
  burst setup and completion boundaries.

## Timing And Utilization

| Metric | PE-pipeline baseline | TileLoader pipeline | Delta |
| --- | ---: | ---: | ---: |
| Post-route WNS | -0.480 ns | +0.089 ns | +0.569 ns |
| Slice LUTs | 23,881 | 23,875 | -6 |
| Slice registers | 21,573 | 21,607 | +34 |
| Block RAM tiles | 5 | 5 | 0 |
| DSPs | 220 | 220 | 0 |

The 100 MHz setup constraint is now met. The worst setup path is local to the
TileLoader, from `col_r[0]` to `burst_len_calc_r[4]`, with 9.737 ns data-path
delay and 16 logic levels. The old coordinate-to-state and coordinate-to-error
paths are absent.

## Board CNN Diagnostic

The same fruit CNN, seed (`2026`), warmup count (`1`), firmware, board, and
100 MHz PL clock were used for both measurements.

| Metric | PE-pipeline baseline | TileLoader pipeline | Delta |
| --- | ---: | ---: | ---: |
| Firmware layer time | 219.465 ms | 224.595 ms | +5.130 ms (+2.34%) |
| Host execute round trip | 231.175 ms | 236.215 ms | +5.040 ms (+2.18%) |
| Host total time | 255.423 ms | 260.468 ms | +5.044 ms (+1.97%) |
| Arena high-water usage | 565,568 B | 565,568 B | 0 |
| Output logits | `[-25, 33, -20]` | `[-25, 33, -20]` | identical |

The latency increase is isolated to the substantial Conv2D layers:

| Layer | Baseline | TileLoader pipeline | Delta |
| ---: | ---: | ---: | ---: |
| Conv2D 0 | 47.893 ms | 49.050 ms | +1.156 ms (+2.41%) |
| Conv2D 4 | 67.332 ms | 69.728 ms | +2.396 ms (+3.56%) |
| Conv2D 8 | 47.113 ms | 48.693 ms | +1.580 ms (+3.35%) |

This is a measured frequency-for-latency tradeoff. Frequency closure is kept;
the next latency optimization should amortize the boundary cycles with longer
Conv2D bursts or prefetch overlap rather than recombining the control cones.

## Rejected Variants

- Direct final-beat advancement recovered boundary cycles but signed off at
  `-0.075 ns`; it was rejected because it did not meet 100 MHz.
- A local replica of the matrix selector shifted the worst route into address
  seed and AvgPool logic and did not produce a stable timing-clean image.

## Validation

- 17/17 active Vivado simulations passed.
- TileLoader exact-cycle checks: activation `22/100`, offset activation
  `23/101`, and weight `28/58` cycles for first-request/completion.
- Integrated GEMM, Conv2D, requantize, activation, pool, upsample, and concat
  simulation passed.
- FPGA manager state: `operating`.
- BudgetCUDA service state: `active`.
- FPGA-manager BIN SHA-256:
  `D91850F64F225F488F234606F8F3FA494B69E7D00C621134E7F56792D868AC6C`.
