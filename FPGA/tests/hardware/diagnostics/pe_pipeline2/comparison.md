# PE Operand Pipeline Comparison

## Change

The INT8 PE now registers both operands before the registered multiply and
registered accumulator stages. The LUT and DSP PE branches retain identical
latency. The GEMM controller and systolic-array testbench include the added
drain cycle.

## Implementation Timing

| Metric | Bursty baseline | PE operand pipeline | Delta |
| --- | ---: | ---: | ---: |
| Post-route WNS | -0.919 ns | -0.480 ns | +0.439 ns |
| WNS violation magnitude | 0.919 ns | 0.480 ns | -47.8% |
| Slice LUTs | 24,162 | 23,881 | -281 (-1.16%) |
| Slice registers | 19,048 | 21,573 | +2,525 (+13.26%) |
| Block RAM tiles | 5 | 5 | 0 |
| DSPs | 220 | 220 | 0 |

The previous systolic PE LUT-multiplier path is no longer the critical path.
The new worst setup path is in `conv2d_tile_loader_ddr`, from replicated
`col_r[0]` control to `state[0]`, with 17 logic levels and 10.336 ns data-path
delay. Timing is improved but is not yet closed at 100 MHz.

## Board CNN Diagnostic

The same fruit CNN, input seed (`2026`), warmup count (`1`), board, firmware,
and 100 MHz PL clock were used for both measurements.

| Metric | Bursty baseline | PE operand pipeline | Delta |
| --- | ---: | ---: | ---: |
| Firmware layer time | 221.046 ms | 219.465 ms | -1.581 ms (-0.72%) |
| Host execute round trip | 223.581 ms | 231.175 ms | +7.595 ms (+3.40%) |
| Host total time | 248.026 ms | 255.423 ms | +7.397 ms (+2.98%) |
| Arena high-water usage | 565,568 B | 565,568 B | 0 |
| Output logits | `[-25, 33, -20]` | `[-25, 33, -20]` | identical |

All three substantial Conv2D layers improved in the firmware measurement:

| Layer | Baseline | PE operand pipeline | Delta |
| ---: | ---: | ---: | ---: |
| Conv2D 0 | 48.151 ms | 47.893 ms | -0.258 ms (-0.54%) |
| Conv2D 4 | 67.752 ms | 67.332 ms | -0.421 ms (-0.62%) |
| Conv2D 8 | 47.555 ms | 47.113 ms | -0.442 ms (-0.93%) |

The host increase is outside PL execution and is consistent with transport and
host scheduling variance in a single Ethernet sample. The firmware layer timer
is the appropriate paired measure for this RTL change. The extra PE cycle does
not cause a measurable CNN latency regression.

## Validation

- 17/17 active Vivado RTL simulations passed.
- PE golden trace, 16x16 systolic array, GEMM core, and 8-array cluster passed.
- FPGA manager state: `operating`.
- 100 MHz FCLK register: `0x00200500`.
- MMIO status remained `0x00012000` across the smoke test.
- Bitstream SHA-256: `1CB147EC505F1FD8DE0B351BED9BAFFC1189FF9E3F83FCB329FD34982AA00CC5`.
- FPGA-manager BIN SHA-256: `445B88614F9469A5DF50303A25C3A5EB29C6DD9E82C33C4293DE9C11E448F973`.
