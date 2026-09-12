# GEMM-only LLM overlay — 2026-09-11

Result: routed timing PASS; BIT/HWH/XSA exported and checked. Not programmed
or tested on the board. RTL source checkpoint: `18c22fb` on `LLM_research`.
Export-script correction is recorded in the following result checkpoint.

## What was removed

The LLM backend no longer instantiates DeiT rowwise LayerNorm/Softmax,
HeatViT pruning/scoring, elementwise residual/GELU, standalone requant/packing,
or multi-operator request arbitration. The shared scheduler's default-on
`ENABLE_FUSED_WRITER` parameter is disabled for LLM, removing its fused
requant/GELU writer at elaboration. The explicit simulation/board source
manifest does not include any of these operator modules. A post-synthesis
forbidden-cell check passed. Original reference sources remain recoverable
and excluded, not physically deleted. No changes to the DeiT branch.

Retained: AXI-Lite, GP0/HP0 bridge, request skid stages, DDR driver/cache,
W4 prefetch/cache, dual 8x8 GEMM cores, command FIFO and completion tags.

## Timing and resources

Vivado 2024.2; xc7z020clg400-1; FCLK0 100 MHz; inherited 0.300 ns setup
uncertainty retained. No clock reduction or new false paths.

| Metric | Previous full backend | GEMM-only |
| --- | ---: | ---: |
| Routed WNS (ns) | -0.394 | +0.125 |
| Routed WHS (ns) | +0.037 | +0.036 |
| Failing setup endpoints | 94 | 0 |
| Routed LUT | 26,691 | 7,486 |
| Routed registers | 30,492 | 8,790 |
| BRAM tiles | 42 | 38 |
| DSP | 165 | 128 |

GEMM-only synthesis: 7,654 LUT / 8,935 registers / 38 BRAM / 128 DSP.
Routed TNS/THS both zero, zero unconstrained internal endpoints or no-clock
pins. DRC has no errors, but retains 256 DSP input-pipeline warnings, nine
slice pairing warnings and 128 DSP configuration advisories. Board-preset
negative DDR skew warnings remain a board-validation consideration.

## Validation and limitations

- `sim/run.ps1`: real DSP48E1 signed W8/W4 tile goldens, tags, input stalls,
  completion backpressure, invalid recovery; FIFO full replacement/wrap/reset.
- `sim/run_integration.ps1`: PASS at 39,080 ns. Mixed queued W8/W4 GEMM/GEMV,
  exact DDR results, independent AW/W, B/R backpressure, memory stalls, full
  rejection, tagged ACK, fault blocking and reset.
- Added two 17x13 multitile cases, including dual cores, N-tile prefetch,
  W4 cache hits/misses and untouched output-row padding. Logical K=9 is
  zero-padded to physical K=10; all 442 outputs match unpadded arithmetic.
- Existing host transformer-op regression: 22 passed.

An initial *un-padded* W4 17x13x9 test returned hardware error 0x4: the second
N tile starts only 4-byte aligned and the retained cache receiver rejects its
partial qword. This general unaligned W4 case remains unsupported. It was not
fixed by operator pruning. The ABI documents 8-byte A/B alignment and even-K
zero padding in both A and B. Padding preserves arithmetic; it does not restore
quantization accuracy. Physical FPGA goldens remain untested.

This is a plain INT32-output W4A8/W8A8 GEMM platform, not full TinyStories
inference, a completed work-stealing scheduler, or a measured T0 model.

## Artifacts and reproducibility

All paths below are relative to `FPGA/llm/`:

- `build/gemm_release/llm_pynq_z2.bit`, `.hwh`, `.xsa`
- `build/gemm_release/{timing_impl,utilization_impl,utilization_synth,hierarchy_synth,drc}.rpt`
- `build/gemm_board/budgetcuda_board.runs/impl_1/bcuda_ps_pl_wrapper_routed.dcp`
- `build/gemm_build.log`: complete synth/route and first BIT/HWH export;
  initial XSA export failed because direct write_bitstream was not registered
  with the implementation run.
- `build/gemm_export.log`: corrected run-managed bitstream and XSA export,
  `LLM_RELEASE_PASS WNS=0.125 WHS=0.036`, process exit 0.

`build_bitstream.tcl` now uses `export_release.tcl`, which checks timing before
running the write_bitstream step and exporting the matched release set.
`resume_export.tcl` reopens the existing routed project for export-only recovery.
The corrected export path was executed; the entire fresh-build script was not
rerun after that export-only correction.

HWH verifies `llm_axi_queued_ip`, base 0x43C80000, FCLK0 100 MHz, HP0 width 64.
The XSA archive includes BIT/HWH/PS initialization. Its embedded BIT SHA256
matches the standalone BIT.

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| BIT | 4045679 | 51ec8f25010a8367ae4daaa95b2ccdccc55125b548ff2be999d3d781b18c9578 |
| HWH | 182025 | 4f71a989bc7eecc120be876036b12b59388d6e52bee13c5415741321e8eb1f39 |
| XSA | 2002637 | c9be9e5c3cfc16167ace74dd9b6b2affae791e57384f8b2b3cbd3713e690a18e |

Build artifacts are ignored local files, not part of the Git checkpoint.
Previous failed build directories and original DeiT BIT/HWH remain intact;
original hashes were rechecked unchanged. No FPGA programming was performed.
