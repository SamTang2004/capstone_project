# DeiT-Tiny full operator-fusion report

## Result

The resident W4A8 DeiT-Tiny graph now fuses every standalone requantization and
activation-layout conversion into its producer. The compiled graph has 328
layers, down from 546:

| Operator | Layers |
| --- | ---: |
| GEMM | 218 |
| Softmax | 36 |
| LayerNorm | 25 |
| Residual add | 24 |
| Token concat | 12 |
| GELU | 12 |
| Token embedding | 1 |
| Standalone requantize | 0 |
| Standalone activation B-pack | 0 |

The resident DDR allocation, including descriptors, is 14,723,066 bytes.

## Fused layouts

- GEMM can write tiled-A, dense row-major with an arbitrary byte stride, or a
  directly consumable dynamic-B layout.
- Attention K projection writes transposed dynamic-B directly.
- Attention V projection writes untransposed dynamic-B directly.
- Softmax writes tiled activations directly for the following context GEMM.
- Token concatenation writes tiled activations directly for output projection.
- The fused writer stores one 8x8 byte tile and selects the required layout at
  its registered write port; it does not retain duplicate row/column copies.

The PYNQ production wrapper omits the three standalone conversion engines that
the fused graph no longer issues. Their RTL and legacy-enabled integrated/unit
tests remain available for compatibility verification.

## Verification

- Host tests: 84 passed, 1 skipped.
- Firmware tests: 62 passed.
- RTL simulations: 25 passed, including tiled, odd-stride row-major, and both
  dynamic-B writer modes.
- Vivado implementation: zero DRC errors and zero unrouted nets.

## PYNQ-Z2 implementation at 100 MHz

| Metric | Result |
| --- | ---: |
| WNS | +0.081 ns |
| WHS | +0.046 ns |
| Setup/hold TNS | 0.000 ns / 0.000 ns |
| Slice LUTs | 13,408 / 53,200 (25.20%) |
| Slice registers | 11,951 / 106,400 (11.23%) |
| BRAM tiles | 39.5 / 140 (28.21%) |
| DSPs | 153 / 220 (69.55%) |

## Physical benchmark

The timing-qualified image was deployed to the PYNQ-Z2 and the complete
resident graph was run from the 64 MiB accelerator arena. Model data remains in
DDR for the full inference; the measured path is host input, device graph, host
output.

| Metric | Result |
| --- | ---: |
| Classification | `apple_fresh` |
| Device logits | `[2, -2, 2, 0, 0, -5]` |
| Reference comparison | Exact match |
| PL cycle window | 116,181,263 cycles |
| PL inference time at 100 MHz | 1.16181263 s |
| Full device-graph stage | 1.2473783 s |
| End-to-end host time | 1.3704571 s |
| Operations | 2,512,612,704 |
| Effective PL throughput | 2.16393 GOPS |
| Busy cycles | 110,347,807 (94.98%) |
| Memory-stall cycles | 49,067,449 (42.23%) |
| Resident model weights | 2,870,904 bytes |
| Total resident allocation | 14,723,066 bytes |

The previous pipelined-writeback image required 204,966,023 cycles. Full
fusion saves 88,784,760 cycles, a 43.3168% reduction and 1.7642x speedup. The
current result is 66,181,263 cycles above the 50,000,000-cycle (500 ms) target;
reaching it still requires a 56.9638% reduction from this implementation.

## Per-module cycle report

Every operation class is included below. Token embedding and concatenation are
firmware operations, so they have a measured graph window but no PL busy-cycle
or completed-operation increment.

| Module | Layers | Previous cycles | Fused cycles | Fused total |
| --- | ---: | ---: | ---: | ---: |
| GEMM | 218 | 60,131,602 | 50,624,076 | 43.57% |
| Softmax | 36 | 30,161,377 | 30,135,188 | 25.94% |
| LayerNorm | 25 | 19,250,176 | 19,202,424 | 16.53% |
| GELU | 12 | 6,908,241 | 6,896,474 | 5.94% |
| Residual add | 24 | 4,824,861 | 4,787,155 | 4.12% |
| Token concat | 12 | 488,870 | 3,465,607 | 2.98% |
| Token embedding | 1 | 1,615,390 | 1,070,339 | 0.92% |
| Standalone activation B-pack | 0 | 46,283,293 | 0 | 0.00% |
| Standalone requantize | 0 | 35,302,213 | 0 | 0.00% |

The three largest remaining modules are GEMM, Softmax, and LayerNorm; together
they account for 86.04% of the total cycle window.

## GEMM phase report

| Phase | Cycles | GEMM share |
| --- | ---: | ---: |
| Load A | 6,340,856 | 12.53% |
| Load B | 17,622,411 | 34.81% |
| Compute | 10,599,617 | 20.94% |
| Write | 15,036,955 | 29.70% |

GEMM is now dominated by loading B and output writeback rather than arithmetic.
The next optimization pass should therefore prioritize B-side bandwidth/cache
reuse and writeback overlap, followed by the Softmax and LayerNorm datapaths.

The machine-readable all-layer report is
[`fpga_w4a8_full_operator_fusion_report.json`](fpga_w4a8_full_operator_fusion_report.json).
