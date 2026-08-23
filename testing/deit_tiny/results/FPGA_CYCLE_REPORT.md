# DeiT-Tiny FPGA cycle report

Measured on the PYNQ board with the complete W8A8 DeiT-Tiny model resident in
the 64 MiB DDR arena. The run used one image write, one fused graph execution,
and one logits read. PL counters run at 100 MHz and are captured as coherent
64-bit values before and after every hardware-backed layer.

## Run result

- Input: `FreshApple/FreshApple (1).jpg`
- Prediction: `apple_fresh`
- Logits: `[2, -2, 1, 2, -1, -7]`
- Model load: 58.648 s (one-time upload)
- Host-observed diagnostic graph execution: 8.521 s
- Host-observed complete inference: 8.605 s
- Exact summed PL execution windows: 838,806,703 cycles, or 8.388067 s
- PL busy: 804,608,732 cycles, or 8.046087 s (95.92% of the window)
- PL memory stall: 493,071,580 cycles, or 4.930716 s (58.78% of the window)
- PL input stall: 108,766,728 cycles, or 1.087667 s
- PL output stall: 190,202 cycles, or 1.902 ms
- Counted operations: 2,527,422,456
- Effective throughput: 0.3014 GOPS
- 8x8 INT8 peak: 12.8 GOPS; measured end-to-end utilization: 2.35%
- Completed hardware operations: 653
- Failed firmware layers: 0; hardware error-latched layers: 0

Stall counters overlap busy cycles and may overlap one another, so they must
not be added to the total execution window.

## Time by operation

| Operation | Layers | Window (ms) | Share | Busy (ms) | Memory stall (ms) |
|---|---:|---:|---:|---:|---:|
| GEMM | 218 | 5,199.90 | 61.99% | 4,896.74 | 3,613.59 |
| LayerNorm | 25 | 1,184.56 | 14.12% | 1,182.68 | 404.94 |
| Requantize | 218 | 810.35 | 9.66% | 797.63 | 511.84 |
| Softmax | 36 | 624.17 | 7.44% | 622.63 | 77.68 |
| Activation-B packing | 120 | 437.11 | 5.21% | 432.06 | 231.23 |
| GELU | 12 | 69.16 | 0.82% | 68.59 | 54.77 |
| Residual add | 24 | 47.17 | 0.56% | 45.75 | 36.67 |
| Token embedding/concat | 13 | 15.66 | 0.19% | 0 | 0 |

The slowest individual hardware operation took 11,345,826 cycles (113.458 ms).
The repeated MLP FC1 layers remain the dominant entries in the top-layer list.

## Interpretation

Relative to the firmware-bias-optimized baseline, the cache-line scheduler,
single-cycle BRAM feed, activation-tile retention, and rowwise parameter cache
reduced the exact window from 8.416317 s to 8.388067 s: 28.25 ms, or 0.34%.
GEMM accounts for 27.08 ms of that reduction. Busy and memory-stall totals each
fell by only about 0.90 ms, showing that the main gain was request/setup idle
time and that repeated B-tile delivery still dominates GEMM.

Removing every reported memory-stall cycle would still leave about 3.46
seconds. Reaching 500 ms requires about 5.05 GOPS, or 39.5% of the 12.8 GOPS
array peak, versus 2.35% today.

The next optimization order should therefore be:

1. Add a second 8x8 core that shares each loaded B tile while processing two M
   tiles, reducing the dominant repeated weight-tile traffic as well as doubling
   useful MAC work per weight delivery.
2. Fuse or queue GEMM, bias, requantization, and packing to remove INT32 DDR
   round trips and reduce 653 hardware launches.
3. Redesign activation-B packing and requantization around wide internal lanes;
   together they consume 1.25 seconds.
4. Pipeline LayerNorm and Softmax arithmetic; cached parameter delivery alone
   cannot improve their compute-dominated 1.81-second total.

The optimized per-layer data is in `fpga_cycle_report_optimized.json`; the
original and firmware-bias-only reports remain beside it for comparison.
