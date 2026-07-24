# Budget-CUDA PL Hardware

This Vivado project implements the current PL target for Budget-CUDA: one tiled INT8 systolic GEMM engine behind AXI-Lite control and an AXI DDR path. The final PYNQ-Z2 bitstream top is `bcuda_tiled_systolic_ddr_top`. The cluster RTL remains parameterizable up to eight engines for a larger FPGA target, but that configuration exceeds the Zynq-7020 resources on PYNQ-Z2.

## Golden Traces

Generate reference traces before simulation:

```powershell
python ..\tests\hardware\create_pe_golden_trace.py
python ..\tests\hardware\create_systolic_golden_trace.py
python ..\tests\hardware\create_gemm_golden_trace.py
python ..\tests\hardware\create_tileloader_golden_trace.py
python ..\tests\hardware\create_tilewriter_golden_trace.py
```

Outputs are written to `FPGA/tests/hardware/golden_traces`.

## Vivado Project

From Vivado Tcl:

```tcl
source FPGA/hardware/vivado/create_project.tcl
```

Default part is `xc7z020clg400-1`, suitable for common Zynq-7020 boards such as PYNQ-Z2/Zybo-class targets. Adjust the part in `vivado/create_project.tcl` if the actual board differs.

Final bitstream project sources are intentionally explicit in `vivado/final_bitstream_sources.tcl`:

- `rtl/axi_lite_control.sv`
- `rtl/memory_io_driver.sv`
- `rtl/tile_loader_ddr.sv`
- `rtl/tile_writer_ddr.sv`
- `rtl/requantize_ddr.sv`
- `rtl/pe_int8_mac.sv`
- `rtl/systolic_array_16x16.sv`
- `rtl/gemm_16x16_core.sv`
- `rtl/gemm_array_cluster_8.sv`
- `rtl/activation_unfold_parallel.sv`
- `rtl/gemm_controller.sv`
- `rtl/accumulator_buffer.sv`
- `rtl/bcuda_tiled_systolic_ddr_top.sv`

Older stream-DMA wrappers have been removed from the prototype tree so they cannot be selected accidentally.

The firmware-visible AXI-Lite register map for this top is documented in `docs/bcuda_ddr_gemm_register_map.md`.

## Board Wrapper

`vivado/create_ps_pl_wrapper.tcl` creates the PYNQ-Z2 PS7/AXI-Lite wrapper and
maps the accelerator register window at `0x43C80000`. The required official
PYNQ-Z2 board files are vendored under `board_files/XilinxBoardStore`; they
provide the DDR/MIO preset. The wrapper now connects the accelerator's
memory-command protocol through a one-outstanding AXI4 bridge to PS HP0 DDR.

On the board, use `budgetcuda.PynqPlRuntime` to allocate PYNQ CMA buffers,
flush A/B before submit, and invalidate C after completion. Its register base
matches the wrapper's `0x43C80000` mapping.

Build the board wrapper, implementation reports, bitstream, and XSA with:

```powershell
vivado -mode batch -source FPGA/hardware/vivado/build_pynq_z2_bitstream.tcl
```

Run all golden-trace hardware simulations:

```powershell
vivado -mode batch -source FPGA/hardware/vivado/run_hardware_tests.tcl
```

Run only the final bitstream top and its required dependencies:

```powershell
vivado -mode batch -source FPGA/hardware/vivado/run_final_top_sim.tcl
```

For manual GUI simulation of the final path, set the simulation top to `tb_bcuda_tiled_systolic_ddr_top`. Other `tb_*` modules are component-level regression benches for blocks still used by the tiled DDR path.

The standalone RTL top is not a board-level pinout. Its AXI-Lite and memory integration ports are intended to connect internally to the final PS/PL wrapper. The project therefore installs `vivado/bitstream_standalone_drc_hook.tcl` as a temporary write-bitstream pre-hook to downgrade unconstrained physical I/O DRCs. Remove that hook once the design is wrapped in a real block design or a board XDC assigns safe package pins.

For direct checkpoint-style bitstream generation during this temporary stage, run:

```powershell
vivado -mode batch -source FPGA/hardware/vivado/write_standalone_bitstream.tcl
```

## Current Memory-Mapped Path

## Tile Loader Design Files

`rtl/tile_loader_ddr.sv` is the DDR-read tile engine. It accepts a base address,
row stride, tile shape, and matrix select; issues one read request per tile
byte; and emits cache write strobes into the systolic core tile memories.

The direct testbench is `sim/tb_tile_loader_ddr.sv`.

`tb_tile_loader_ddr` checks the read address sequence and cache writes against `tileloader_reads.hex`, `tileloader_A_tile.hex`, and `tileloader_B_tile.hex`.

## Tile Writer Design Files

`rtl/tile_writer_ddr.sv` is the DDR-write tile engine. It accepts a C base address, row stride, tile shape, and an INT32 tile read port; then emits one write request per output element.

The direct testbench is `sim/tb_tile_writer_ddr.sv`. It checks the write address sequence and data against `tilewriter_writes.hex`, then compares the simulated DDR image against `tilewriter_final_ddr.hex`.

The top reads INT8 A/B tiles from DDR through `memory_io_driver`, computes with
the 16x16 systolic core, accumulates partial K tiles in `accumulator_buffer`,
and writes INT32 C tiles through `tile_writer_ddr`.

## PYNQ-Z2 Programming

`vivado/build_pynq_z2_bitstream.tcl` emits the board bitstream and XSA. For a
Linux FPGA-manager image, convert the Vivado `.bit` file with:

```powershell
python FPGA/hardware/vivado/pynq_bit2bin.py FPGA/hardware/bcuda_pynq_z2.bit FPGA/hardware/bcuda_pynq_z2_fpga_manager.bin
```

The resulting file contains the raw headerless, word-swapped payload expected
by the PYNQ-Z2 kernel's `/sys/class/fpga_manager/fpga0/firmware` interface.
