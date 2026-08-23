# DeiT-Tiny FPGA cycle report

Measured on the PYNQ board with the complete W8A8 DeiT-Tiny model resident in
the 64 MiB DDR arena. The run used one image write, one fused graph execution,
and one logits read. PL counters run at 100 MHz and are captured as coherent
64-bit values before and after every hardware-backed layer.

## Run result

- Input: `FreshApple/FreshApple (1).jpg`
- Prediction: `apple_fresh`
- Logits: `[2, -2, 1, 2, -1, -7]`
- Model load: 57.662 s (one-time upload)
- Host-observed diagnostic graph execution: 9.050 s
- Exact summed PL execution windows: 890,446,943 cycles, or 8.904469 s
- PL busy: 804,705,337 cycles, or 8.047053 s (90.37% of the window)
- PL memory stall: 493,167,821 cycles, or 4.931678 s (55.38% of the window)
- PL input stall: 108,773,895 cycles, or 1.087739 s
- PL output stall: 190,202 cycles, or 1.902 ms
- Counted operations: 2,527,422,456
- Effective throughput: 0.2839 GOPS
- 8x8 INT8 peak: 12.8 GOPS; measured end-to-end utilization: 2.22%
- Completed hardware operations: 653
- Failed firmware layers: 0; hardware error-latched layers: 0

Stall counters overlap busy cycles and may overlap one another, so they must
not be added to the total execution window.

## Time by operation

| Operation | Layers | Window (ms) | Share | Busy (ms) | Memory stall (ms) |
|---|---:|---:|---:|---:|---:|
| GEMM | 218 | 5,716.51 | 64.20% | 4,897.62 | 3,614.48 |
| LayerNorm | 25 | 1,184.58 | 13.30% | 1,182.72 | 404.97 |
| Requantize | 218 | 810.21 | 9.10% | 797.64 | 511.85 |
| Softmax | 36 | 624.16 | 7.01% | 622.63 | 77.68 |
| Activation-B packing | 120 | 437.12 | 4.91% | 432.09 | 231.27 |
| GELU | 12 | 69.19 | 0.78% | 68.60 | 54.77 |
| Residual add | 24 | 47.16 | 0.53% | 45.75 | 36.67 |
| Token embedding/concat | 13 | 15.53 | 0.17% | 0 | 0 |

The slowest individual layer was block 11 MLP FC1: 14,419,940 window
cycles (144.199 ms), including 11,412,171 busy cycles and 8,424,068 memory
stall cycles. The repeated MLP FC1 layers dominate the top-layer list.

## Interpretation

GEMM is the largest target, but the result is not simply a DDR bandwidth
problem. Removing every reported memory-stall cycle would still leave about
3.97 seconds. Non-GEMM primitives consume 35.8% of the measured window, and
GEMMs include about 819 ms where the PL is not busy. Reaching 500 ms requires
about 5.05 GOPS, or 39.5% of the 12.8 GOPS array peak, versus 2.22% today.

The next optimization order should therefore be:

1. Increase DDR burst efficiency and cache/reuse weight and activation tiles,
   especially for MLP GEMMs.
2. Fuse or queue GEMM, requantization, and packing to remove per-operation
   setup gaps and reduce 653 hardware launches.
3. Stream and pipeline LayerNorm, Softmax, and requantization; together they
   consume 2.62 seconds.
4. Only add a second 8x8 array after memory delivery and scheduling can keep
   one array substantially better utilized.

The complete per-layer data is in `fpga_cycle_report.json`.
