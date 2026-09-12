# LLM research progress — 2026-09-11

## Verified results

- GEMM-only PYNQ-Z2 build: BIT/HWH/XSA generated at 100 MHz; routed WNS
  +0.125 ns, WHS +0.036 ns. 7,486 LUT, 8,790 registers, 38 BRAM, 128 DSP.
  Old DeiT operators excluded; AXI/DDR/cache and dual GEMM cores retained.
  See `vivado/GEMM_BUILD_RESULT.md` for artifact hashes and build limitations.
- Exact sparse/dense row-fusion compiler: 143 tasks -> 50 tiles; actual DSP48E1
  XSim checks pass for 3,200 tile values and 1,049 final task outputs after
  software scatter/merge. Maximum absolute error 0. Dense-fused control is
  57 tiles, versus 330 unfused; most savings are fusion, not sparsity alone.
- Independent-dot diagonal RTL adapter: requested outputs 11 and 83 returned
  under their original lane tags. Measured execution cycles: raw separate 16,
  compacted separate 12, compacted fused 6, padded K=16 fused 20. Three input
  stalls increase fused execution to 9. Eight-lane signed and backpressure
  checks pass. Actual operand/completion trace is tracked at
  `research/diagonal_trace.jsonl`; report: `research/DIAGONAL_RTL_RESULT.md`.
- Research Python suite: 41 tests pass. Includes pre-wave S*T0 budget gating,
  exact sparsity golden checks, repeated steal-cost experiments, trace corruption
  rejection and deterministic fixture generation.

## What is NOT yet demonstrated

- No new image programmed onto the FPGA; no board latency or power measurements.
- No RTL sparse detector/compactor, hardware compiler, or measured S*T0 deadline.
- Diagonal adapter is not in the production board manifest; it has not been
  synthesized or timing-qualified. Off-diagonal cross products remain wasteful.
- Fusion/compiler/transport cost is excluded from the 16/12/6/20-cycle comparison.
- No demonstrated win against independently scheduled dot-product lanes or a
  strong sparse accelerator baseline. No full TinyStories inference speedup.
- Dynamic activation sparsity is only knowable before an input-ready compute
  wave, not before all future layers/tokens. Approximate sparsity needs separate
  model-quality evaluation; the reported fusion goldens use exact zeros only.

## Next decision gate

Build an independent dot-product lane baseline with comparable arithmetic,
SRAM and bandwidth resources. Then include sparse classification, compaction,
tag/accumulator routing and transport in end-to-end timing. Compare dense-fused,
simple sparse, independent-lane and proposed scheduling modes. Measure latency,
useful MAC occupancy, area/Fmax and energy; do not substitute tile-count or
execution-window speedup for system-level improvement.

Working targets discussed (project goals, not industry standards): >=20%
improvement against a strong sparse baseline for a worthwhile research lead;
>=1.5x end-to-end for a competitive prototype, with 2x as an ambitious product
candidate target or a corresponding energy/cost advantage. Novelty and product
readiness require more than any single speedup number.

## Saved milestones and reproduction

- `42410fa`: timing-qualified GEMM-only release and result record.
- `9414324`: repeated work-stealing/T0 controls.
- `481c230`: pre-wave sparse fusion planner.
- `e207c95`: mixed sparse/dense compiler goldens and actual RTL trace validation.
- `146ccf7`: diagonal RTL result routing and measured cycle comparison.

```powershell
& FPGA/llm/sim/run_diagonal.ps1
& FPGA/llm/sim/run_fusion.ps1
.\.venv\Scripts\python.exe -m pytest FPGA/llm/research -q -p no:cacheprovider
```

Large model/golden/build artifacts remain local and are not included in this
progress update. Generators, tests, source code, reports and the small diagonal
trace are versioned. The prior pre-DeiT-reset research checkpoint is preserved
in Git ancestry at d8c6abe. The original DeiT branch and images are unchanged.
