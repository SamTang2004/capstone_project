# DeiT-Tiny FPGA operator-fusion report

This report compares the same six-class W4A8 DeiT-Tiny model before and after
fusing GEMM bias and requantization into tiled INT8 writeback. Both captures
come from the physical PYNQ-Z2 at 100 MHz. The fused multiplier uses registered
DSP48 input operands and a registered DSP product; the final implementation
passes timing with WNS +0.001 ns and WHS +0.051 ns.

## Result

The fused graph preserves the reference logits `[2, -2, 2, 0, 0, -5]`. Its
exact PL window falls from 272,895,347 to 217,613,870 cycles, a 20.26% reduction
from 2.729 to 2.176 seconds. Host-observed graph execution falls from 2.853 to
2.269 seconds (20.48%). This remains 4.35x above the 500 ms target.

Fusion contracts the graph from 666 to 546 layers by absorbing 120 of 218
requantization layers into Q/K/V and MLP-FC1 GEMM writeback. It also removes
the corresponding INT32 accumulation buffers: reported arena use falls from
41,559,184 to 28,813,840 bytes.

| Module | Fused layers | Before cycles | After cycles | After time | Change |
|---|---:|---:|---:|---:|---:|
| GEMM, including fused writeback | 218 | 85,712,932 | 73,407,830 | 734.08 ms | -14.36% |
| Activation-B packing | 120 | 43,682,944 | 46,123,120 | 461.23 ms | +5.59% |
| Standalone requantization | 98 | 80,903,476 | 35,332,873 | 353.33 ms | -56.33% |
| Softmax | 36 | 30,147,701 | 30,174,739 | 301.75 ms | +0.09% |
| LayerNorm | 25 | 19,261,349 | 19,253,971 | 192.54 ms | -0.04% |
| GELU | 12 | 6,912,267 | 6,920,543 | 69.21 ms | +0.12% |
| Residual add | 24 | 4,709,098 | 4,833,076 | 48.33 ms | +2.63% |
| Token embedding | 1 | 1,080,049 | 1,082,358 | 10.82 ms | +0.21% |
| Token concatenation | 12 | 485,531 | 485,360 | 4.85 ms | -0.04% |

Small changes in modules untouched by fusion are run-to-run DDR contention and
arbitration effects. Module stall counters overlap their module windows and
must not be added to the times above.

## GEMM phases after fusion

| GEMM phase | Cycles | Time |
|---|---:|---:|
| Load A | 6,352,465 | 63.52 ms |
| Load B | 17,244,944 | 172.45 ms |
| Systolic compute | 10,599,617 | 106.00 ms |
| Fused/ordinary writeback | 29,417,342 | 294.17 ms |

The fused path eliminates a large round trip through DDR and 120 standalone
requant launches, but direct packed writeback itself is now expensive. The
largest remaining whole-model consumers are GEMM (33.73%), activation packing
(21.19%), standalone requantization (16.24%), Softmax (13.87%), and LayerNorm
(8.85%). The next optimization should therefore target write/pack bandwidth
and the 98 row-major requantization layers, rather than adding arithmetic cores.

Raw captures are `fpga_w4a8_cached_report.json` and
`fpga_w4a8_fused_report.json` in this directory.
