# FPGA W4A8 BRAM-cache report

## Result

The deployable DeiT-Tiny path now uses signed W4 matrix weights and signed INT8
activations/accumuland inputs. Packed W4 weights remain resident in DDR3 for
the model lifetime. Each GEMM launch loads a static weight tile once into a
72 KiB scheduler-local BRAM cache; later M-tile pairs reuse it without another
DDR weight request.

The 100 MHz PYNQ-Z2 image meets timing with WNS **+0.008 ns** and WHS
**+0.025 ns**. The complete design uses **39.5 BRAM tiles**. All 24 RTL golden
simulations pass, including a 20-row W4 scheduler test that checks the second
M pair produces no repeated B request.

## Physical comparison

Both rows use the same input, resident graph, 100 MHz clock, and W4A8 model.
The cached run reproduced the uncached logits exactly: `[2, -2, 2, 0, 0, -5]`.

| Counter | Uncached W4A8 | BRAM-cached W4A8 | Change |
|---|---:|---:|---:|
| Exact PL window | 3.5761 s | 2.7290 s | -23.7% |
| GEMM window | 1.7023 s | 0.8571 s | -49.6% |
| GEMM B-load | 951.5 ms | 148.5 ms | -84.4% |
| GEMM A-load | 61.1 ms | 61.0 ms | -0.1% |
| GEMM compute | 106.0 ms | 106.0 ms | unchanged |
| GEMM write | 241.2 ms | 241.2 ms | unchanged |
| Effective model throughput | 0.707 GOPS | 0.927 GOPS | +31.1% |

The unchanged compute and write counters isolate the improvement to static W4
weight delivery. The remaining 148.5 ms of B-load includes the first fill of
each GEMM launch and dynamic attention matrices, which are INT8 and cannot use
the static W4 cache.

## Current module bottlenecks

| Rank | Module | Exact cycles | Time | PL share |
|---:|---|---:|---:|---:|
| 1 | GEMM | 85,712,932 | 857.1 ms | 31.41% |
| 2 | Requantize | 80,903,476 | 809.0 ms | 29.65% |
| 3 | Activation-B pack | 43,682,944 | 436.8 ms | 16.01% |
| 4 | Softmax | 30,147,701 | 301.5 ms | 11.05% |
| 5 | LayerNorm | 19,261,349 | 192.6 ms | 7.06% |
| 6 | GELU | 6,912,267 | 69.1 ms | 2.53% |
| 7 | Residual add | 4,709,098 | 47.1 ms | 1.73% |

At 2.729 s, another 5.46x reduction is required for the 500 ms target. The next
architectural target is the separate GEMM-output/requantization pass, followed
by activation transpose/packing. Fusing GEMM accumulation, bias, requantization,
and tile-major INT8 output would attack both the 809 ms requantizer and the
241 ms GEMM-write phase while reducing DDR traffic.

Raw reports: `fpga_w4a8_uncached_report.json` and
`fpga_w4a8_cached_report.json`.
