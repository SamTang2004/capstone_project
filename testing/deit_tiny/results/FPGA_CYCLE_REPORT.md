# DeiT-Tiny FPGA cycle report

Measured on the PYNQ board with the complete W8A8 DeiT-Tiny model resident in
the 64 MiB DDR arena. Each run used one image write, one device-resident graph
execution, and one logits read. PL counters run at 100 MHz.

## Dedicated rowwise-engine experiment

The shared reciprocal LayerNorm/Softmax engine was built with timing signoff
(WNS +0.045 ns, WHS +0.027 ns) and measured on the complete resident model.

| Metric | Dual-core baseline | Rowwise build | Change |
|---|---:|---:|---:|
| Exact PL window | 4.517346 s | 3.143402 s | 30.42% lower |
| Host complete inference | 4.708 s | 3.348 s | 28.89% lower |
| LayerNorm window | 723.81 ms | 192.76 ms | 73.37% lower |
| Softmax window | 527.78 ms | 301.53 ms | 42.87% lower |
| Effective throughput | 0.5597 GOPS | 0.8044 GOPS | 43.72% higher |

The initial rowwise build shifted logits to `[2, -3, -1, 2, -3, -8]`. Prefix-
graph captures localized the first mismatch to the first LayerNorm: its gamma
and beta BRAMs were read one column late after the first output. The standalone
engine test did not expose this because it bypassed the DDR wrapper, while the
wrapper test used the same gamma for every column. Prefetching `col_idx + 1` on
an accepted output and testing distinct per-column gamma values fixed the
alignment. The corrected first LayerNorm is byte-for-byte identical to the
dual-core baseline (SHA-256 `937e4348...a87c52e`), and the complete graph now
restores exact logits `[2, -2, 1, 2, -1, -7]`.

Two corrected diagnostic runs measured 3.197357 and 3.143402 seconds in the
exact PL window. The table uses the latest run; the optimized architecture's
observed reduction is 29.22% to 30.42% relative to the same dual-core baseline.

## Corrected rowwise-build cycles by module

| Module | Layers | Cycles | Window (ms) | Total window | Busy cycles | Memory-stall cycles |
|---|---:|---:|---:|---:|---:|---:|
| GEMM | 218 | 127,079,860 | 1,270.80 | 40.43% | 97,372,652 | 65,147,851 |
| Requantize | 218 | 80,981,440 | 809.81 | 25.76% | 79,773,069 | 51,193,493 |
| Activation-B pack | 120 | 43,700,354 | 437.00 | 13.90% | 43,209,362 | 23,127,014 |
| Softmax | 36 | 30,153,217 | 301.53 | 9.59% | 30,003,992 | 11,300,876 |
| LayerNorm | 25 | 19,276,397 | 192.76 | 6.13% | 19,097,424 | 4,829,174 |
| GELU | 12 | 6,919,935 | 69.20 | 2.20% | 6,859,888 | 5,477,272 |
| Residual add | 24 | 4,713,525 | 47.14 | 1.50% | 4,575,219 | 3,667,011 |
| Token embedding | 1 | 1,067,163 | 10.67 | 0.34% | 0 | 0 |
| Token concat | 12 | 448,344 | 4.48 | 0.14% | 0 | 0 |

Busy and stall counters overlap the module window and each other. They identify
why a module is slow but must not be added to derive elapsed time.

The next target is a fused GEMM-output path. Requantization and activation
packing together consume 124,681,794 cycles (39.66% of the complete run), with
74,320,507 memory-stall cycles between them. Producing tiled INT8 directly from
the GEMM accumulator can eliminate the intermediate INT32 DDR write/read and
can avoid a separate pack pass. GEMM remains the largest individual category,
but 65,147,851 of its cycles are also memory stalls, so a third arithmetic core
alone would leave the dominant data-movement problem in place. At graph level,
the 12 attention stages consume 196,532,385 cycles (62.52%) and the 12 MLP
stages consume 113,404,428 cycles (36.08%). The largest individual layers are
the MLP `fc1` GEMMs and their immediately following requantizers, making that
pair the first implementation and measurement target for the fused path.

The corrected raw data is in `fpga_cycle_report_row_engine_fixed.json`.
Boundary captures for the baseline, faulty build, and corrected build are in
the three `discrepancy_*.json` files.

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
