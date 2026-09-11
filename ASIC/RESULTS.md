# ASIC exploration results

These are exploratory OpenROAD results using the open Nangate45 standard-cell
library and the predictive ASAP7 platform. They are useful for architecture
comparison, but they are not foundry signoff or tapeout-ready results.

## Nangate45 versus ASAP7

The ASAP7 experiment uses the same RTL, parameters, and 100 MHz interface
constraint as Nangate45. It uses the predictive ASAP7 7.5-track RVT library at
the typical corner. ASAP7 reports timing in picoseconds; the 10,000 ps period,
2,000 ps I/O delay, and 100 ps uncertainty are equivalent to the Nangate45
10 ns, 2 ns, and 0.1 ns constraints.

| Metric | Nangate45 post-route | ASAP7 post-global-route |
| --- | ---: | ---: |
| Synthesis cell area | 245,966 um^2 | 18,257 um^2 |
| Functional area at reported stage | 253,420 um^2 | 20,261 um^2 |
| Die area | 0.607 mm^2 | 0.0711 mm^2 |
| Utilization | 46% | 33% |
| Setup / hold violations | 0 / 0 | 0 / 0 |
| Estimated minimum period | 4.73 ns | 4.507 ns |
| Estimated Fmax | 211.61 MHz | 221.88 MHz |
| Vectorless power at 100 MHz | 20.8 mW | 3.76 mW |
| Global-route wire length | - | 1.525 m |
| Detailed-route DRC | 0 | Not converged |

The ASAP7 area is about 12.5x smaller and its vectorless power estimate about
5.5x lower at the same requested clock. Its estimated Fmax is only about 5%
higher in this experiment, which says that this RTL's mapped critical path and
I/O budget matter more than node name alone. These are not process-scaling
factors: the libraries, corners, routing models, floorplans, and completion
stages differ.

ASAP7 CTS required roughly 16,300 hold buffers because the very fast cells
created many minimum-delay paths at a relaxed 100 MHz target. A 30% initial
floorplan closed global-route timing and electrical limits, but detailed route
still began with about 50,600 local rule conflicts and used about 25 GB of
memory. The run was stopped after confirming poor convergence, so no ASAP7 GDS
or post-route extraction is claimed.

## Routed dual-GEMM boundary

The routed target is `transformer_gemm_scheduler`: two 8x8 INT8 GEMM arrays
(128 MAC processing elements), their scheduler, caches, and writeback logic.
Arithmetic width and array size are unchanged. The inferred scratchpads use
reduced exploration depths (`MAX_K=16`, `MAX_N=8`, and
`MAX_W4_CACHE_VECTORS=8`) because the open Nangate45 platform has no SRAM
compiler.

| Metric | Result |
| --- | ---: |
| Clock constraint | 10.00 ns (100 MHz) |
| Final setup / hold violations | 0 / 0 |
| Post-route minimum period | 4.73 ns (211.61 MHz) |
| Post-route setup slack at 100 MHz | +5.27 ns |
| Die size | 779.32 x 779.32 um (0.607 mm^2) |
| Functional-cell area | 253,420 um^2 (0.253 mm^2) |
| Final utilization | 46% |
| Sequential cells | 13,027 |
| Detailed-route DRC violations | 0 |
| Antenna violations | 0 |
| Routed wire / vias | 3.317 m / 1,013,746 |
| Vectorless estimated power | 20.8 mW |
| Worst VDD / VSS IR drop | 0.765 mV / 0.677 mV |
| Final GDS size | 127,750,864 bytes |

The timing and geometry checks pass, but the final report still contains 148
maximum-capacitance violations. Those need electrical repair before this can be
called clean. Power is OpenROAD's default vectorless activity estimate, not a
workload-annotated measurement.

Timing-driven global placement is disabled for this target because this
OpenROAD revision crashes in its FLUTE estimation path on the design. Timing
repair remains enabled after placement and routing, and the extracted final
timing is clean at the requested 100 MHz.

## Full production top

The full `bcuda_transformer_ddr_top` parses and reaches Yosys memory analysis.
It infers 38 memories totaling 963,220 bits (about 118 KiB); the largest is
9,216 x 64, or 589,824 bits. Lowering these memories into standard-cell flops
is neither representative nor tractable, so the full configuration stops at a
4,096-bit memory guard. A meaningful full-chip run requires SRAM macros, LEF,
Liberty timing models, and a macro-placement/power plan for the selected
foundry process.

## Reproduction

From the repository root in PowerShell:

```powershell
.\ASIC\run_openroad.ps1 -Design dual-gemm -Target finish -Engine wsl
.\ASIC\run_openroad.ps1 -Design full -Target synth -Engine wsl -Clean
```

The second command is intentionally expected to stop at the memory guard after
printing the inventory.
