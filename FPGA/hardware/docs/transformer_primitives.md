# Transformer vector primitives and debug contract

The vector path uses INT8 activations. LayerNorm gamma and beta are signed Q8
INT16 values, Softmax produces unsigned Q0.7 probabilities in an INT8-sized
buffer, and GEMM/bias accumulation remains INT32.

## Implemented blocks

- `layernorm_int8`: vector statistics and streamed per-channel gamma/beta.
- `gelu_int8`: eight Q4.4 piecewise-quadratic lanes.
- `softmax_int8`: max subtraction, Q15 exponential LUT, Q0.7 output.
- `residual_add_int8`: eight saturating lanes with alignment shifts.
- `bias_add_int32`: eight parallel INT32 bias additions.
- `transformer_debug_counters`: cycle, stall, operation, error, and trace state.
- `transformer_control_status`: maps counters into the AXI-Lite debug bank.

## Debug register bank

The old `0x00..0x7c` ABI remains intact. The Transformer extension is:

| Address | Meaning |
| ---: | --- |
| `0x80` | enable, step, freeze, clear in bits 0..3 |
| `0x84..0xb8` | seven low/high 64-bit cycle counters |
| `0xbc` | completed operations |
| `0xc0` | active op, sticky error, trace-valid |
| `0xc4` | trace event |
| `0xc8` | trace data |

Firmware exposes `bcuda_axi_dma_debug_control` and
`bcuda_axi_dma_read_transformer_debug`. Python exposes
`PynqPlRuntime.set_debug()` and `debug_snapshot()`. Graph diagnostics preserve
the v2 record size; Transformer hardware slots are exposed through
`LayerDiagnostics.transformer_cycles`.

## PYNQ-Z2 synthesis checkpoint

| Block | LUT | FF | DSP |
| --- | ---: | ---: | ---: |
| Debug counters | 183 | 590 | 0 |
| Residual add | 312 | 0 | 0 |
| Bias add | 256 | 0 | 0 |
| GELU | 2,968 | 0 | 24 |
| LayerNorm | 8,384 | 146 | 13 |
| Softmax | 1,288 | 62 | 1 |

These are isolated synthesis results on `xc7z020clg400-1`, not additive
placed-design results. GELU is currently combinational and the divider paths
still require top-level pipelining and timing closure.

Protocol opcodes 12–15 append LayerNorm, GELU, Softmax, and residual add without
renumbering legacy operations. Firmware CPU fallbacks and the Python mock match
the RTL integer rules. `build_deit_mlp_block_package` builds a pre-norm MLP
residual block; QKV transpose/head scheduling is the next full-encoder step.
