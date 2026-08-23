# DeiT-Tiny dual-core FPGA cycle report

Measured on the PYNQ board with the complete W8A8 DeiT-Tiny model resident in
the 64 MiB DDR arena. Each run used one image write, one device-resident graph
execution, and one logits read. PL counters run at 100 MHz.

## Run result

- Input: `FreshApple/FreshApple (1).jpg`
- Prediction: `apple_fresh`
- Logits: `[2, -2, 1, 2, -1, -7]` (identical to the one-core result)
- Model load: 49.913 s (one-time upload)
- Host-observed graph execution: 4.534 s
- Host-observed complete inference: 4.708 s
- Exact summed PL execution windows: 451,734,614 cycles, or 4.517346 s
- PL busy: 418,527,077 cycles, or 4.185271 s (92.65% of the window)
- PL memory stall: 189,029,509 cycles, or 1.890295 s (41.85% of the window)
- Counted operations: 2,527,422,456
- Effective end-to-end throughput: 0.5597 GOPS
- Two-core 8x8 INT8 peak: 25.6 GOPS; end-to-end utilization: 2.19%
- Completed hardware operations: 653
- Failed firmware layers: 0

Stall counters overlap busy cycles and may overlap one another, so they must
not be added to the total execution window.

## Time by operation

| Operation | Layers | Window (ms) | Busy (ms) | Memory stall (ms) |
|---|---:|---:|---:|---:|
| GEMM | 218 | 1,887.06 | 1,592.94 | 1,041.28 |
| Requantize | 218 | 809.93 | 797.65 | 511.86 |
| LayerNorm | 25 | 723.81 | 721.95 | 5.75 |
| Softmax | 36 | 527.78 | 526.27 | 8.69 |
| Activation-B packing | 120 | 437.10 | 432.09 | 231.26 |
| GELU | 12 | 69.16 | 68.60 | 54.77 |
| Residual add | 24 | 47.19 | 45.77 | 36.69 |
| Token embedding/concat | 13 | 15.33 | 0 | 0 |

## One-core comparison

| Metric | One core | Two cores | Change |
|---|---:|---:|---:|
| Exact PL window | 8.388067 s | 4.517346 s | 46.15% lower; 1.857x speedup |
| GEMM window | 5.199897 s | 1.887057 s | 63.71% lower; 2.756x speedup |
| GEMM busy | 4.896740 s | 1.592941 s | 67.47% lower |
| GEMM memory stall | 3.613590 s | 1.041278 s | 71.18% lower |
| Effective throughput | 0.3014 GOPS | 0.5597 GOPS | 85.71% higher |
| Largest hardware operation | 113.458 ms | 36.305 ms | 68.00% lower |

Two post-deployment runs produced exact PL windows of 4.581327 s and 4.517346
s. Both produced identical logits. The table uses the second, saved run; the
observed range is a 45.38% to 46.15% reduction relative to the one-core
8.388067 s baseline.

The direct architectural effect is clearest in GEMM: paired M tiles share one B
tile load and execute concurrently, reducing the GEMM window by 3.313 seconds.
LayerNorm and Softmax also measured below the earlier run even though their RTL
was not changed here, so those differences should be treated as board/run-state
variation rather than attributed to the second GEMM core.

GEMM remains the largest single category, followed by requantization,
LayerNorm, Softmax, and activation packing. Reaching 500 ms still requires
removing about 4.02 seconds. The next priority is graph-level fusion/queuing to
avoid INT32 DDR round trips, followed by wider requantization and activation
packing and pipelined LayerNorm/Softmax arithmetic.

The full per-layer dual-core data is in `fpga_cycle_report_dual_core.json`.
The one-core comparison source is `fpga_cycle_report_optimized.json`.
