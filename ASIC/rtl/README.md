# ASIC RTL deployment snapshot

This directory is a deployment copy of `FPGA/hardware/rtl`.  ASIC flows must
use the files here so technology-specific changes do not affect the FPGA build.

The GEMM scheduler's five logical 64-bit memories instantiate
`bcuda_asap7_sram_1rw_64`, which banks the ASAP7 `fakeram7_128x64` hard macro.
OpenROAD receives the matching Verilog black box, LEF, and Liberty views from
`ASIC/openroad/dual_gemm_asap7/config.mk`.

`(* ram_style = "block" *)` is a Vivado inference hint, not an ASIC macro-mapping
directive.  A production foundry flow should keep the wrapper interface and
replace its implementation and physical views with macros from that foundry's
memory compiler.

The supplied ASAP7 macro is single-port.  This copy therefore disables the W4
background prefetch that can overlap cache reads with writes; foreground loads
and cache reuse are unchanged.  If a dual-port compiled SRAM becomes available,
the overlap can be restored in this ASIC copy.

The full ASAP7 top also macro-backs the large rowwise sample, gamma, beta,
residual, output-tile, and HeatViT score stores.  Narrow memories use 512x8 or
256x32 macro slices; the byte-written output tile uses eight independent byte
lanes.  Small control FIFOs, the 256-entry exponential table, HeatViT weights,
and package accumulators remain standard-cell memories because each is at or
below the configured 4-Kbit synthesis threshold or requires many simultaneous
reads.

## Full-top ASAP7 result (2026-08-26)

Run the deployment with:

```powershell
.\ASIC\run_openroad.ps1 -Design full -Platform asap7 -Target finish -Engine wsl
```

The completed 100 MHz flow retained 131 hard SRAM instances: 99
`fakeram7_128x64`, 31 `fakeram_512x8`, and one `fakeram7_256x32`.  The final
591.653 um square die contains 86,784 um2 of logic and macros at 27.5% core
utilization.  Global routing has zero overflow; detailed routing converged from
141,123 initial conflicts to zero DRC violations after seven optimization
passes.  The detailed router peaked at about 31 GB of memory, so a 64 GB host
has adequate RAM for this configuration.

Post-route OpenRCX timing meets setup by 5.338 ns and reports a modeled maximum
frequency of 214.5 MHz.  It still reports 315 hold violations (worst -96.9 ps,
TNS -594.6 ps), 2,638 maximum-slew violations, and three maximum-capacitance
violations.  These are signoff cleanup items even though routing DRC is clean.
Estimated power is 63.7 mW at 0.7 V; worst VDD drop is 0.429 mV and worst VSS
bounce is 0.441 mV.

The public FakeRAM release has LEF and Liberty views but no SRAM GDS.  KLayout
therefore emits the final GDS with the `fakeram.*` cells intentionally empty.
The ODB and DEF contain the macro footprints and routed interfaces, but a real
tapeout must substitute foundry memory-compiler GDS/CDL views and rerun timing,
DRC, and LVS.
