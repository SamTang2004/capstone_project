# LLM PYNQ-Z2 bitstream build

Vivado 2024.2, xc7z020clg400-1, existing vendored PYNQ-Z2 board preset.
Run from the repository root (Windows subprocess/CScript access is required):

```powershell
& 'C:/Xilinx/Vivado/2024.2/bin/vivado.bat' -mode batch -source FPGA/llm/vivado/build_bitstream.tcl -log FPGA/llm/build/vivado_build.log -journal FPGA/llm/build/vivado_build.jou
```

Create `FPGA/llm/build` first if needed. Do not run two builds in the same
directory. The create script regenerates only `FPGA/llm/build/gemm_board`; it does
not touch `FPGA/hardware/vivado_board_project` or the DeiT BIT/HWH files.

`create_board.tcl` is derived from the DeiT board script with the same PS7,
GP0 interconnect, 64-bit HP0 bridge, reset topology and address assignment.
The only module-reference substitution is `llm_axi_queued_ip`. That wrapper
uses the same AXI/memory port metadata as the original IP wrapper. Backend
`llm_gemm_ddr_top` includes only GEMM and the retained DDR/cache machinery.
Its scheduler sets `ENABLE_FUSED_WRITER=0`; no DeiT operators are compiled.
`rtl_sources.txt` is shared with integration simulation. Synthesis additionally
checks for forbidden operator cells and writes `hierarchy_synth.rpt`.
The supported queued commands remain plain W8/W4 GEMM, not an LLM decoder.

Default FCLK is 100 MHz; `LLM_FCLK_MHZ` can select an explicitly requested
alternative. `LLM_BUILD_JOBS` defaults to 4. The original 0.300 ns setup clock
uncertainty constraint is retained. Reports and release outputs go under
`FPGA/llm/build/gemm_release`. The build checks setup and hold slack before writing
`llm_pynq_z2.bit`, pairs it with the generated HWH, and exports XSA. A timing
failure stops release generation; no device is programmed by this script.

Board-preset warnings about negative PS DDR DQS-to-clock delays are inherited
from the vendored preset and must be noted during board validation, not hidden.
Production-parameter RTL integration regression: `sim/run_integration.ps1`.

Expected outputs after success:

- `llm_pynq_z2.bit`, `llm_pynq_z2.hwh`, `llm_pynq_z2.xsa`
- `utilization_synth.rpt`, `utilization_impl.rpt`, `timing_impl.rpt`, `drc.rpt`
- console marker `LLM_RELEASE_PASS` with WNS/WHS (clock is recorded in HWH)

Latest validated result: `GEMM_BUILD_RESULT.md`. Export-only recovery from the
existing routed `build/gemm_board` project is available through
`resume_export.tcl`; it repeats the timing gate before run-managed bitstream
generation. The full build uses the same `export_release.tcl` helper.

Bitstream generation is not a physical functional test. The board still
requires identity/readback, golden results, and clock verification before
performance measurements. The old daemon must not run against the new ABI.
