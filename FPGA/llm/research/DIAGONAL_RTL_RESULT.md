# Independent sparse dot products: actual RTL verification

Vivado 2024.2 XSim with real DSP48E1: PASS, eight commands, 512 full matrix
outputs checked, plus per-lane tag/result routing. Simulation ends at 1370 ns.
No board programming, synthesis or timing closure of this adapter was performed.

Inputs from the requested example:

```
tag 100: [1,2,1,0] dot [3,4,0,0] = 11
tag 101: [5,0,6,0] dot [7,0,8,0] = 83
```

W8 mode is used because positive 8 is outside signed W4's range. The compacted
two input pairs are prepared in the testbench; no RTL zero detector/compiler
or compaction latency is claimed. In fused mode:

```
step 0: row A=[1,5], column B=[3,7]
step 1: row A=[2,6], column B=[4,8]
matrix result=[[11,23],[39,83]]
```

`llm_diagonal_tile.sv` is a new RTL adapter around the unchanged tagged core.
It captures up to eight 32-bit job tags and a lane count at command acceptance.
Each diagonal PE owns one job for the entire command. The adapter extracts
PE[i,i] into result lane i with its original tag and valid mask; cross products
are discarded. Values/tags are emitted in parallel under one ready/valid
handshake, not sequentially over an AXI bus. The combinational extraction adds
no registered cycle in this simulation; its physical timing/resources have
not been measured. Per-element tags and interleaved accumulator contexts are
not required by this fixed-ownership mode.

## Measured execution cycles

| Mode | Commands | K per command | Execution cycles | Requested results |
| --- | ---: | ---: | ---: | --- |
| Original independent jobs | 2 | 4 | 8+8=16 | 11,83 |
| Compacted independent jobs | 2 | 2 | 6+6=12 | 11,83 |
| Compacted diagonal fusion | 1 | 2 | 6 | 11,83 |
| Fused, padded to K=16 | 1 | 16 | 20 | 11,83 |
| Fused plus 3 input-stall cycles | 1 | 2 | 9 | 11,83 |
| Eight signed independent dots | 1 | 2 | 6 | -25,-16,-9,-4,-1,0,-1,-4 |

All cycle counts were asserted against both the core wrapper counter and the
actual difference between equally sampled command/completion clock edges.
This removes the mixed pre/post-NBA tick convention in the earlier fusion
trace. K+4 applies to wrapper completion, with K input steps, clear, two drain
steps and wrapper capture. At the testbench's 100 MHz clock, 16/12/6/20 cycles
represent 160/120/60/200 ns of counted execution. These are not FPGA measurements.
Command gaps, four-negative-edge completion backpressure, upstream compaction,
DDR, routing/merge into external buffers and bus transfer are excluded. The
two-command figures are sums of execution windows, not continuous wall time.

Savings are separated: compacting both dots removes four input steps total
(16 -> 12); issuing them together shares the six-cycle command (12 -> 6).
Fixed-K padding adds fourteen useless input steps (6 -> 20). DSP removal was
not necessary to eliminate those steps: existing variable-K control sufficed.

The current 8x8 array broadcasts operands rather than forwarding a systolic
wavefront between PEs. Independent dot fusion remains spatially wasteful:
two desired diagonal outputs among four active cross products in this example;
eight desired diagonal outputs among 64 for eight jobs. Fewer execution cycles
does not imply high useful MAC utilization or low energy. Off-diagonal PEs
remain in the original datapath; this adapter does not disable them.

## Correctness and reproducibility

The bench checks the entire matrix, including 23/39 cross results and inactive
zeros, then the independently routed diagonal values/tags. It changes command
tags, tile ID, batch and lane count after acceptance to test snapshot ownership.
All result values, tags and masks stay stable with completion backpressure.
An additional eight-lane signed case exercises every diagonal extraction lane.

Run `FPGA/llm/sim/run_diagonal.ps1` from the repository root. The script checks
compile/elaboration exit status and the explicit XSim PASS marker. Actual
operand-handshake and completion events are in
`FPGA/llm/build/diagonal/diagonal_trace.jsonl`; this result's trace is also
checkpointed beside this document as `diagonal_trace.jsonl`.

The adapter is experimental and deliberately absent from the board source
manifest. The existing LLM bitstream is unchanged. Before hardware deployment,
add format/descriptor ingress, real sparse compaction, bounded ownership queues
and transport accounting; then synthesize and validate their end-to-end cost.
