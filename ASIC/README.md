# ASIC exploration

This directory translates the production BudgetCUDA RTL with OpenROAD Flow
Scripts and the open Nangate45 or predictive ASAP7 standard-cell libraries. The default measurable
implementation is `transformer_gemm_scheduler`: both 8x8 GEMM arrays (128
INT8 MACs total), their scheduler, cache control, and fused writeback logic.

The `BCUDA_ASIC` compile define makes two elaboration-time changes:

- Xilinx `DSP48E1` MACs become equivalent pipelined standard-cell multipliers.
- Legacy standalone operators omitted by the production FPGA wrapper remain
  omitted in the ASIC build.

The runner defaults to the project-local Ubuntu WSL toolchain:

```powershell
.\ASIC\run_openroad.ps1 -Target synth -Clean
.\ASIC\run_openroad.ps1 -Target finish
.\ASIC\run_openroad.ps1 -Platform asap7 -Target grt -Clean
```

Use `-Engine docker` for the ORFS container. Use `-Design full` to parse the
complete `bcuda_transformer_ddr_top` and produce its inferred-memory inventory.
That configuration deliberately stops when it reaches an array larger than
4 Kbit: the full RTL contains 38 inferred memories, including a 9,216x64-bit
store, and Nangate45 has no compiled SRAM views. Flattening those arrays into
flip-flops is neither representative nor computationally useful.

For the ASAP7 dual-GEMM experiment, scratchpad depths are reduced at
elaboration (`MAX_K=16`, `MAX_N=8`, `MAX_W4_CACHE_VECTORS=8`) and the ASIC RTL
copy explicitly instantiates banked `fakeram7_128x64` macros. The matching
Liberty and LEF views are supplied by the ASAP7 configuration. A production
foundry integration should preserve the wrapper boundary and replace the
predictive FakeRAM implementation with memory-compiler outputs.

Generated logs, reports, objects, and results are written beneath
`ASIC/OpenROAD-flow-scripts/flow/{logs,reports,objects,results}/<platform>/`.

The ASAP7 configuration uses the predictive 7.5-track RVT library at its
typical corner and picosecond timing units. Its 10,000 ps SDC period is the
same 100 MHz constraint as Nangate45's 10 ns period. Use `-Target grt` for the
repeatable post-global-route result. The earlier standard-cell-only detailed
route did not converge; the SRAM-macro deployment is kept under the distinct
`dual_gemm_asap7_sram` design nickname so its results do not overwrite that
experiment.
