# Lookahead TileLoader CNN Benchmark

Date: 2026-07-26

## Configuration

- Board: PYNQ-Z2, Zynq-7020
- PL clock target: 100 MHz
- Datatype: packed signed INT4
- Tile path: four-bank lookahead BRAM cache
- CNN benchmark iterations: 100
- Routed timing before post-route physical optimization: WNS -0.382 ns
- Bitstream: `bcuda_pynq_z2.bit`, generated 2026-07-26 13:57:59

## Repeated Inference

| Metric | Result |
|---|---:|
| Model upload | 0.521 s |
| Mean latency | 328.097 ms |
| Median latency | 328.767 ms |
| P95 latency | 339.073 ms |
| Maximum latency | 340.122 ms |
| Total inference time | 32.810 s |
| Software-device equality | 100/100 |
| ONNX tolerance pass rate | 47% |
| ONNX class agreement | 18% |
| Worst ONNX error | 8.296408 |

## Firmware Diagnostics

| Layer | Operator | Previous ms | Lookahead ms | Latency reduction | Throughput gain |
|---:|---|---:|---:|---:|---:|
| 0 | Conv2D GEMM | 54.640 | 29.169 | 46.6% | 1.87x |
| 4 | Conv2D GEMM | 290.415 | 117.050 | 59.7% | 2.48x |
| 8 | Conv2D GEMM | 286.710 | 113.097 | 60.6% | 2.54x |
| 14 | Classifier GEMM | 0.079 | 0.029 | 63.6% | 2.74x |

- Previous summed firmware layer time: 651.145 ms.
- Lookahead summed firmware layer time: 278.686 ms.
- Overall firmware-layer reduction: 57.2%.
- Overall firmware-layer speedup: 2.34x.
- Diagnostic host total: 315.421 ms.
- Diagnostic firmware total: 278.828 ms.
- Conv2D GEMM share of firmware time: 93.0%.
- Pooling share of firmware time: 6.9%.

## Resource Snapshot

| Resource | Used | Device capacity | Utilization |
|---|---:|---:|---:|
| Slice LUTs | 26,552 | 53,200 | 49.91% |
| Slice registers | 21,730 | 106,400 | 20.42% |
| BRAM tiles | 1.5 | 140 | 1.07% |
| DSPs | 50 | 220 | 22.73% |

## Interpretation

The lookahead path removes most of the previous Conv2D loading penalty. The two
large convolution layers are approximately 2.5 times faster, while pooling
latency remains effectively unchanged. End-to-end INT4 latency is still 47.1%
slower than the recorded 223 ms INT8 run, so the next performance work should
target Conv2D execution below the cache, particularly array feed utilization
and remaining serialized lowering.

The FPGA output matched the BudgetCUDA software device for all 100 runs, but
agreement with ONNX Runtime remains poor. That isolates the remaining accuracy
problem to the shared INT4 quantization/lowering semantics rather than random
hardware corruption.
