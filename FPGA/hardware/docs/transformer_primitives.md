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
| `0x84..0xb8` | seven low/high global cycle counters |
| `0xbc` | completed operations |
| `0xc0` | active op, sticky error, trace-valid |
| `0xc4` | trace event |
| `0xc8` | trace data |
| `0xcc..0xd0` | GEMM B-weight cycles, low/high |
| `0xd4..0xd8` | GEMM bias-load/cache-read cycles, low/high |
| `0xdc` | W4 tile-cache hits |
| `0xe0` | W4 tile-cache misses |

Firmware exposes `bcuda_axi_dma_debug_control` and
`bcuda_axi_dma_read_transformer_debug`. Python exposes
`PynqPlRuntime.set_debug()` and `debug_snapshot()`. Graph diagnostics preserve
the v5 record size; Transformer hardware slots are exposed through
`LayerDiagnostics.transformer_cycles`.

## Pipelined W4 Load B

W4 cache-line requests no longer wait for the previous 64-byte response before
issuing the next. The GEMM scheduler keeps the memory driver's four-entry
ordered request queue populated, then launches a background fill for N tile
`i+1` while both 8x8 arrays consume tile `i`. Cache storage is layer-wide, so
later M passes continue to reuse every fetched W4 tile without DDR traffic.

Each N tile's optional eight-element INT32 bias is cached as four 64-bit words
in a synchronous BRAM. Later M passes spend only the short BRAM read sequence
instead of issuing four small DDR reads. The aggregate `gemm_load_b_cycles`
counter is retained, with separate weight and bias counters plus tile hit/miss
counts added to diagnostics version 5.

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
# Optional HeatViT token pruning

`TOKEN_PRUNE` (opcode 19) is an ABI-compatible fused selector. It consumes a
tiled INT8 token matrix and an INT8 or W4 `[hidden_dim, 1]` scorer, preserves
CLS, retains a fixed number of highest-scoring patch tokens, restores their
original sequence order, and optionally appends the rounded mean of all removed
tokens as a package token. Its fixed output cardinality allows every following
GEMM descriptor to use smaller static `M`/`N` dimensions while selection remains
image-adaptive.

Layer fields are `input0=tokens`, `weight=scorer`, optional `bias=INT32[1]`,
`output=compacted tokens`, `n=input token count`, `c=hidden dimension`, and
`out_c=output token count`. Flag bit zero enables the package token.

The host manifest may contain a `heatvit` object with ordered selectors:

```json
{
  "heatvit": {
    "enabled_by_default": false,
    "selectors": [
      {"after_block": 2, "output_tokens": 151,
       "scorer": "heatvit.selectors.0.score", "package_token": true},
      {"after_block": 5, "output_tokens": 106,
       "scorer": "heatvit.selectors.1.score", "package_token": true},
      {"after_block": 8, "output_tokens": 44,
       "scorer": "heatvit.selectors.2.score", "package_token": true}
    ]
  }
}
```

`DeiTManifestRunner.run_image(..., heatvit=True|False)` selects the pruned or
baseline resident graph. Switching modes replaces activation workspaces and the
descriptor before the next inference; an inference itself remains exactly one
host-to-device graph command and one device-to-host result transfer. A request
for HeatViT is rejected when no selector package is present.

`heatvit_token_scorer_4lane.sv` provides a four-lane registered dot-product path.
`heatvit_token_pruner_ddr.sv` adds the complete DDR-backed selector, repeated
top-K search, source-order compaction, and package-token mean. Control bit 23
dispatches this operation through the production transformer top; the firmware
programs the input/scorer/output addresses, token dimensions, W4/W8 mode, and a
Q16 reciprocal for package averaging. Row-major graphs use this PL path, while
the bit-exact firmware implementation remains the fallback for tiled layouts.

On `xc7z020clg400-1`, isolated OOC synthesis at 100 MHz reports 3,282 LUTs,
744 registers, 9 DSP48E1s, no block RAM, and +0.311 ns setup slack. The complete
transformer top reports 18,445 LUTs (34.67%), 13,042 registers (12.26%), 39.5
BRAM tiles (28.21%), and 171 DSPs (77.73%). Its remaining -2.531 ns OOC path is
between the existing fused GEMM writer and rowwise engine; the token-pruner
itself meets the clock target. These are synthesized, unplaced OOC results and
are not a substitute for post-route timing or board measurements.

The board build accepts `BCUDA_FCLK_MHZ` while retaining 100 MHz as its default.
The first HeatViT bring-up image was built with `BCUDA_FCLK_MHZ=75.0`; the Zynq
PLL generated a 76.929 MHz timing clock (12.999 ns, HWH value 76,923,080 Hz).
Post-route signoff passed with +0.545 ns setup slack and +0.035 ns hold slack,
zero unrouted nets, and zero DRC errors. Implemented utilization is 16,150 LUTs
(30.36%), 12,474 registers (11.72%), 39.5 BRAM tiles (28.21%), and 162 DSPs
(73.64%). This image is intended for functional bring-up before restoring and
closing the 100 MHz target.

The subsequent 100 MHz Load-B-prefetch image closes the same conservative
0.300 ns uncertainty constraint at +0.027 ns setup and +0.009 ns hold, with no
unrouted nets or DRC errors. It uses 18,056 LUTs (33.94%), 13,362 registers
(12.56%), 40.5 BRAM tiles (28.93%), and 162 DSPs (73.64%).
