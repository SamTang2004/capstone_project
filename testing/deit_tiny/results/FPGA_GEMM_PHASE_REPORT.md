# DeiT-Tiny FPGA GEMM phase report

This report is from the timing-clean 100 MHz PYNQ-Z2 overlay with two 8x8
INT8 GEMM cores. The six-class W8A8 graph produced the expected logits
`[2, -2, 1, 2, -1, -7]`. The raw counter capture is in
`fpga_gemm_phase_report.json`.

## Result

The complete PL window was 314,535,720 cycles (3,145.36 ms). GEMM consumed
127,389,460 cycles (1,273.89 ms, 40.50% of the model), but only 10,599,617
cycles (106.00 ms) were arithmetic. GEMM is primarily a data-movement and
fusion problem, not a systolic-array throughput problem.

| GEMM phase | Cycles | Time | Share of active GEMM phases |
|---|---:|---:|---:|
| Load B / weights | 56,488,283 | 564.88 ms | 58.05% |
| Write INT32 output | 24,117,154 | 241.17 ms | 24.78% |
| Compute | 10,599,617 | 106.00 ms | 10.89% |
| Load A / activations | 6,102,737 | 61.03 ms | 6.27% |
| Scheduler overhead while busy | 49,453 | 0.49 ms | 0.05% |

The phase sum is 97,307,791 cycles. The remaining 30,032,216 GEMM window
cycles (300.32 ms) occur while the accelerator is not busy. Firmware currently
adds GEMM bias to the INT32 matrix on the ARM after every applicable GEMM.
Using the no-bias score/context layers as the launch-and-poll control floor
attributes approximately 282 ms of this interval to the ARM bias loops. The
new diagnostics ABI records bias time directly; that field still needs one
fresh board capture because the board became unreachable before the
instrumented rerun completed.

Memory-stall cycles total 65,132,445 (651.32 ms). They overlap the phase rows
above and therefore must not be added to the GEMM window a second time.

## GEMM role breakdown

| Role | Layers | Window | Load A | Load B | Compute | Write | Non-busy |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLP FC1 | 12 | 387.04 ms | 3.13 ms | 157.66 ms | 29.50 ms | 72.77 ms | 123.83 ms |
| MLP FC2 | 12 | 246.87 ms | 12.47 ms | 155.62 ms | 28.94 ms | 18.19 ms | 31.60 ms |
| Attention score | 36 | 113.22 ms | 3.19 ms | 41.03 ms | 8.07 ms | 57.73 ms | 3.07 ms |
| Attention Q | 36 | 108.74 ms | 9.39 ms | 39.32 ms | 7.38 ms | 18.19 ms | 34.43 ms |
| Attention V | 36 | 108.69 ms | 9.39 ms | 39.31 ms | 7.38 ms | 18.19 ms | 34.38 ms |
| Attention K | 36 | 108.65 ms | 9.39 ms | 39.32 ms | 7.38 ms | 18.19 ms | 34.34 ms |
| Attention projection | 12 | 100.55 ms | 3.13 ms | 39.39 ms | 7.38 ms | 18.19 ms | 32.42 ms |
| Attention context | 36 | 79.07 ms | 9.88 ms | 40.25 ms | 7.56 ms | 18.19 ms | 3.15 ms |
| Patch embedding | 1 | 20.77 ms | 1.04 ms | 12.97 ms | 2.41 ms | 1.51 ms | 2.84 ms |
| Classifier | 1 | 0.29 ms | 0.01 ms | 0.01 ms | 0.00 ms | 0.00 ms | 0.27 ms |

MLP FC1 and FC2 alone consume 633.91 ms, almost half of all GEMM time. Their
weight matrices account for 313.28 ms of B loading. Attention score is instead
write-heavy because it materializes the 197x197 INT32 score matrix.

Scheduled lane utilization is about 94% for all major shapes, so adding another
array without fixing memory traffic would attack only the 106 ms compute slice
and increase pressure on the same DDR path.

## Optimization order

1. Fuse bias and requantization into GEMM writeback. Emit packed INT8 directly
   instead of writing INT32 to DDR, processing it on ARM/REQUANTIZE, then
   reading it again. This targets about 241 ms of GEMM writes, about 282 ms of
   ARM bias, and a substantial portion of the separate 808.95 ms requantizer.
2. Add double-buffered B prefetch and persistent weight-tile caching. Weight
   loading is the largest active phase at 564.88 ms; the MLP matrices are the
   first target. Prefetch should overlap the next B tile with current compute.
3. Keep A streaming and overlap it with B/compute where dependencies allow.
   A loading is only 61.03 ms, so it is secondary.
4. Re-measure with the mandatory phase and bias counters after every change.
   Do not add a third general GEMM array until compute becomes a significant
   share of the window.

Even eliminating GEMM entirely would leave about 1.87 seconds in the current
graph, so the 500 ms model target also requires the planned fused requant path
and reductions in activation packing, softmax, and normalization traffic.
