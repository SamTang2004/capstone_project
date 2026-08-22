# Deterministic Ordered Loader Comparison

Reference: native 64-bit HP0 checkpoint at 100 MHz.

| Metric | Reference | Ordered loader | Change |
| --- | ---: | ---: | ---: |
| 100-run mean latency | 83.921 ms | 80.601 ms | -3.96% |
| 100-run p95 latency | 89.156 ms | 87.093 ms | -2.31% |
| 300-image mean latency | 83.363 ms | 81.536 ms | -2.19% |
| Firmware layer time | 52.497 ms | 52.019 ms | -0.91% |
| Main-Conv total cycles | 2,674,124 | 2,626,345 | -1.79% |
| Main-Conv Load A cycles | 1,200,495 | 1,152,740 | -3.98% |
| Main-Conv loader address cycles | 1,147,741 | 1,093,640 | -4.71% |
| FPGA balanced accuracy | 99.00% | 99.00% | unchanged |
| FPGA/ONNX class agreement | 99.33% | 99.33% | unchanged |

The loader change removes measurable address and activation-load work, but
Conv2D still accounts for about half of firmware layer time. Requantization and
pooling account for about another 39%, so further end-to-end gains require a
balanced streaming path across these operators rather than loader-only work.

The deployed image closed timing at 100 MHz with `+0.008 ns` WNS and
`+0.017 ns` WHS. A cold-boot activation probe captured `ARLEN=7`, `ARSIZE=3`,
eight accepted read beats, and `RRESP=OKAY`.
