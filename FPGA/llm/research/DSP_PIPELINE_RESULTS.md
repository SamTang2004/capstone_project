# Full DSP multiplier pipeline experiment

This is the historical DSP-only experiment at 100 MHz. The subsequent
[200 MHz implementation](PIPELINE_200_RESULTS.md) also pipelines selection logic
and specializes the tiled core, superseding the clock limit reported here.

The eight DSP48E1 multipliers now use both operand registers and the multiplier
and output registers: **AREG=2, BREG=2, MREG=1, PREG=1**. Vivado checks these
properties on every synthesized DSP. The product pipeline increases from one to
four stages; the reduction and INT32 accumulator stages are unchanged.

The ALUMODE, INMODE, OPMODE and carry control registers are enabled as well.
C, D and AD registers remain disabled because the multiply-only datapath does
not use the C operand or the pre-adder. Enabling those unused branches would
not pipeline any active arithmetic path. Cascade register attributes follow
AREG/BREG; cascade outputs are unused.

Input enable captures A1/B1 only. Subsequent stages continue clocking through
bubbles so isolated final operands cannot get stuck. Four aligned stages carry
valid, mask, context slot and last flags. Reset clears all in-flight validity.
The interface still accepts one eight-lane beat per cycle, and holds completed
results under backpressure. This experiment makes no power-saving claim.

## Routed 100 MHz comparison

Same xc7z020clg400-1 device, registered timing fixture, 10 ns clock, 0.300 ns
uncertainty, and Explore implementation flow. These are individual routed
results, not a multi-seed statistical comparison or a measured maximum frequency.

| Metric | Previous one-stage product | Four-stage product |
| --- | ---: | ---: |
| Consumer LUTs | 2497 | 2497 |
| Consumer fabric FFs | 1509 | 1542 |
| DSP48E1s | 8 | 8 |
| BRAM36-equivalent tiles | 2.5 | 2.5 |
| Whole-design setup slack | +0.305 ns | +0.198 ns |
| Whole-design hold slack | +0.010 ns | +0.051 ns |
| Worst setup slack from DSP cells | +1.014 ns | +1.951 ns |

The DSP-to-reduction path gains 0.937 ns of slack in these runs. Placement and
routing also change, so this is not an isolated intrinsic DSP delay measurement.
The worst whole-engine path remains outside the DSPs: previously finder keep
bits to the stream descriptor; now the remaining-block bitmap to the registered
activation-request last flag. Thus this change does **not** demonstrate an
increase in whole-engine clock rate. A separately routed 125 MHz trial is
**failing**, with setup slack **-1.186 ns** and hold slack **+0.050 ns** after
post-route physical optimization. The critical path again runs from the
remaining-block bitmap to the activation-request last flag. This is a failed
target, not a qualified 125 MHz implementation. The next timing optimization
should register block selection/request generation; it is outside this DSP-only
experiment.

Resource counts exclude 1072 preserved timing-fixture FFs. DSP internal registers
do not count as fabric FFs. Producer hardware is unchanged. As before, physical
operand memories, response queues, DDR/DMA and board IO are outside this fixture.

## Latency cost

K=4096, 64 output rows, eight lanes, eight activation bytes/cycle, nonzero W8
weights, resident metadata, block skipping and descriptor reuse enabled.

| Valid activation density | Previous cycles | New cycles | New useful MAC utilization |
| --- | ---: | ---: | ---: |
| 100% | 32836 | 32860 | 99.72% |
| 50% | 16452 | 16476 | 99.44% |
| 10% | 3440 | 3464 | 94.46% |
| 1% | 574 | 598 | 53.51% |
| 0.1% | 121 | 145 | 22.07% |
| 0% | 66 | 90 | 0% |

Each of the eight tiles drains three additional cycles, adding 24 cycles.
At 100 MHz, the 0.1% case increases from 1.21 us to 1.45 us. Its break-even
frequency is 119.84 MHz: 100 MHz times 145/121. Merely enabling more registers
therefore makes this short workload slower unless the clock increases enough
or tile drains are overlapped. Dense workloads pay a much smaller relative cost.
Stalled tests can have different cycle deltas because stalls depend on cycle phase.

The separate metadata-building pass still costs 513 cycles; it is not hidden
by this change. Latest matched-baseline latency, memory traffic and resource
tables are regenerated in [TILED_ACTIVATION_RESULTS.md](TILED_ACTIVATION_RESULTS.md).

## Verification and reproduction

- New real-DSP simulation exhaustively checks all 65,536 signed INT8 operand
  pairs with inserted bubbles, reset at all four pipeline depths, and immediate
  post-reset input. It verifies the four-cycle product timing.
- Original sparse-dot regression: 34 cases, including INT32 wraparound, tagged
  reductions, empty completion, stalls and cancellation.
- Pipelined wide-finder regression: 104 cases across scan widths and supply limits.
- Shared-activation regression: all three variants, 129 vectors / 768 dots.
- Tiled regression: 76 commands / 336 tile results / 2548 dots, including a valid
  tile before intentional reset; 32,768 exhaustive producer cases also pass.
- The 100 MHz report validator checks routed setup/hold, internal timing coverage,
  routing errors, DRC severity, resource counts and the two outer-fixture exceptions.

```powershell
& FPGA/llm/sim/run_dsp_pipeline.ps1
& FPGA/llm/sim/run_sparse_dot.ps1
& FPGA/llm/sim/run_wide_sparse_dot.ps1 -PipelineMerge
& FPGA/llm/sim/run_activation_sharing.ps1
& FPGA/llm/sim/run_tiled_activation.ps1
# Default 100 MHz flow and synthesis/report commands: TILED_ACTIVATION.md.
& 'C:/Xilinx/Vivado/2024.2/bin/vivado.bat' -mode batch -source FPGA/llm/vivado/implement_tiled_activation.tcl -log FPGA/llm/build/dsp_pipeline_125.log -journal FPGA/llm/build/dsp_pipeline_125.jou -tclargs 8.000 tiled_activation_dsp_125
```

Local baseline source, results and routed checkpoint were preserved under
`build/dsp_pipeline_baseline` before editing. `vivado/report_dsp_baseline.tcl`
extracts the original DSP path reports from that checkpoint. Historical earlier
experiment tables retain their original pipeline measurements.
