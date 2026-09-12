# First real Vivado board implementation — 2026-09-11

Historical failed full-backend run. The later GEMM-only build passes timing
and exports BIT/HWH/XSA; see `GEMM_BUILD_RESULT.md`. This evidence is retained.

**Result: implemented/routed, but not timing-qualified; no release bitstream.**

Source checkpoint: `5ff949a` on `LLM_research`. Vivado 2024.2 built the real
PYNQ-Z2 PS7/GP0/64-bit HP0 block design with `llm_axi_queued_ip` at
`0x43c80000`. HWH confirms FCLK0=100 MHz and HP0 width=64. Shared DeiT RTL and
Vivado files remain unchanged; the separate LLM IP wrapper disables legacy
standalone operations exactly as the old production wrapper did.

The first sandboxed invocation could not start CScript workers. Its scoped
processes were stopped, and the elevated retry completed synthesis and routing.
The production-parameter queued-AXI/DSP integration regression passed again.

## Saved routed result

- 100 MHz; original setup clock uncertainty 0.300 ns retained.
- WNS **-0.394 ns**, TNS -5.461 ns, 94 failing setup endpoints.
- WHS **+0.037 ns**, zero failing hold endpoints.
- Worst reported path: backend rowwise engine
  `p2_quotient_product0` -> `p3_corrected_reg[32]`.
- Post-implementation: 26,691 LUTs (50.17%), 30,492 registers (28.66%).
- Synthesized resource counts: 42 BRAM tiles, 165 DSPs.
- Routed DRC summary contains warnings/advisories, not errors. DSP pipelining
  warnings remain; inherited board-preset PS DDR skew warnings were not hidden.

Artifacts under the ignored `FPGA/llm/build` directory:

- `release/timing_impl.rpt`, `release/utilization_impl.rpt`, `release/drc.rpt`
- `board/budgetcuda_board.runs/impl_1/bcuda_ps_pl_wrapper_routed.dcp`
- HWH at `board/budgetcuda_board.gen/sources_1/bd/bcuda_ps_pl/hw_handoff/bcuda_ps_pl.hwh`
- `vivado_build_retry.log`

The timing gate correctly stopped before writing BIT/HWH release or XSA.
There was no board programming, no frequency reduction and no relaxed timing
constraint. Previous DeiT bitstream files were not overwritten.

## Post-route recovery attempt

`recover_postroute.tcl` opened the saved routed DCP and ran
`phys_opt_design -directive AggressiveExplore`. Logs showed intermediate
WNS -0.119 ns / WHS +0.037 ns. However, the process exited with code 1 before
writing a final report or optimized DCP. The captured log contains no explicit
reason for that exit. Do not report the intermediate slack as signed-off or
recoverable implementation evidence. `postroute_recovery.log` preserves it.

## Next implementation change

The current compatibility wrapper keeps full DeiT rowwise/HeatViT circuitry,
despite exposing only plain GEMM commands. Introduce an LLM-only elaboration
configuration that removes unused operators while retaining the original
AXI slave, HP0 bridge, memory-driver cache/burst behavior and GEMM scheduler.
Keep the default DeiT configuration unchanged. Re-run RTL goldens, synthesize,
route and apply the same timing gate. The scheduler also has near-critical
prefetch/cache paths, so pruning alone is not a guarantee of 100 MHz closure.
