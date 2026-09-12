# Sparse/dense fusion compiler: verified golden set

Result: PASS. 8 deterministic cases, 143 original W4A8 GEMV tasks, 50 compiled
tiles. XSim executes the real DSP48E1 tile implementation. All 3,200 physical
INT32 tile outputs match; software tag scatter/fragment accumulation produces
1,049 final task outputs identical to independent, unfused dense dot products.
Maximum absolute error: 0. No tasks or outputs are lost or duplicated.

## Actual checks

| Case | Tasks | Final scalars checked | Unfused tiles | Dense fused tiles | Compiled sparse tiles | Max error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Aligned sparsity | 24 | 192 | 48 | 6 | 3 | 0 |
| Disjoint sparsity | 16 | 128 | 32 | 4 | 4 | 0 |
| Mixed sparse/dense rows | 25 | 200 | 75 | 12 | 12 | 0 |
| Dense control | 17 | 136 | 34 | 6 | 6 | 0 |
| All-zero elision | 9 | 72 | 18 | 4 | 0 | 0 |
| Odd K and 3-column tail | 19 | 57 | 57 | 9 | 9 | 0 |
| Weight/batch isolation | 24 | 192 | 48 | 12 | 12 | 0 |
| Signed W4/A8 extrema | 9 | 72 | 18 | 4 | 4 | 0 |
| Total | 143 | 1049 | 330 | 57 | 50 | 0 |

Counts of unfused/dense-fused tiles are analytical controls, not separate RTL
timing runs. 330 -> 50 mainly reflects row fusion; sparsity itself improves
57 -> 50. A dense row can fill the retained-K union and erase sparse-K savings.
An all-zero task requires a software-tagged zero completion, not a zero-K RTL
command. The test explicitly verifies these outputs even though no tile runs.

## Compiler and execution contract

`fusion_goldens.py` generates fixed-seed input weights and activations, obtains
an exact-sparsity plan from `batch_planner.py`, then emits executable fixed
8x8x16 tiles. Threshold=0 removes only zero contributions. This is not an
approximate sparsification/accuracy experiment. Dense here means activation-
dense; random W4 matrices may themselves contain zero scalar weights.

Compatible rows share batch and exact weight identity. Union-K gather uses the
same K order for all lanes. Missing row contributions, K tails and invalid
output lanes are zero padded. K=33 and K=18 inputs become even physical K=16
fragments, avoiding the previously identified odd-K packed-tile boundary.
An immutable tile descriptor maps each lane to the original (batch,task) tag;
the RTL wrapper carries (batch,tile_id), NOT separate tags for every row.
Software performs row scatter and fragment accumulation before comparing with
the original dense reference. Repeated task IDs across different batches and
incompatible weight families are included.

Independent dense goldens sum original A[k]*B[k,c] directly, without the
compiler's retained masks, packed operands or fragment ordering. RTL outputs
are compared against fragment goldens and then independently against final
dense task outputs. All fixture sums fit INT32. General overflow behavior,
requantization, QAT logits and end-to-end TinyStories accuracy remain outside
this set. The compiler is a software prototype, not a synthesized scheduler.

## Execution trace and timing

The actual XSim trace contains tile ID, batch, launch/completion tick,
execution_cycles, input_wait_cycles and all 64 output values. The software trace
contains tile K-index selection, original row tags, partial results and final
job-complete events; its event order is NOT a measured cycle timeline.

All 50 RTL tiles report execution_cycles=20, input_wait_cycles=0 for K0=16.
The bench clock is 10 ns; these are RTL simulation cycles, not board timings.
The wrapper counter excludes command acceptance and counts RUN edges through
result capture. launch_tick/completion_tick have a separate sampling boundary;
do not equate their difference or AXI replay time with this counter. Every
completion is held under backpressure for three negative edges and checked
stable. Total testbench simulation ends at 12,700 ns, including reset, command
gaps and completion holds; it is not a fused-inference latency measurement.

The compiler uses a deliberately permissive S=4096 functional-test budget and
the earlier analytical T0=64; these do NOT establish the S*T0 deadline or match
the newly observed 20-cycle wrapper counter. No planner timing, SRAM capacity,
DDR/cache transport, DMA, AXI queue integration or physical work stealing was
validated in this run. Do not infer system speedup from tile-count reduction.

## Reproduce and inspect

From the repository root:

```powershell
& FPGA/llm/sim/run_fusion.ps1
.\.venv\Scripts\python.exe -m pytest FPGA/llm/research -q -p no:cacheprovider
```

The complete script passed with exit 0, including software generation, Vivado
2024.2 XSim, all RTL tile checks and end-to-end tag verification. The Python
suite has 41 passing tests, including deterministic compilation and rejection
of missing/duplicate tiles, corrupt partial sums, wrong batch and corrupt dense
goldens. Parser unit fixtures use synthetic traces explicitly; the result above
comes from the separately executed XSim testbench.

Generated artifacts in `FPGA/llm/build/fusion_goldens/`:

- `program.json`: all original tasks/weights, dense goldens, classification,
  row/tag/K-index maps, fragment goldens and software final outputs.
- `vectors.txt`: complete packed A8/W4 operands and fragment golden values for
  the testbench; one command header, 16 operand pairs, 64 expected values/tile.
- `software_trace.jsonl`: compiler execution/scatter reference events.
- `rtl_trace.jsonl`: actual XSim tile results and cycle counters.
- `verified_results.json`: per-task RTL-derived final outputs, counts and hashes.
- `xsim.log`: `PASS: fusion golden RTL tiles=50` at 12,700 ns.

Generated files remain local, ignored build artifacts. The deterministic
generator, testbench, runner, unit tests and this result record are checkpointed.
No existing bitstreams or FPGA configuration were changed.

SHA256 of this run:

| File | SHA256 |
| --- | --- |
| program.json | d84c8bdd47aafb559d30424b2a576529bddee273065c26554e298c554747a47f |
| vectors.txt | 927812a4ac35e17bc661b9083f8504f947375536027ef0ab9e35637e760195af |
| software_trace.jsonl | 8f63e61015b88ae2289e98934714ac36116603d258ae03b8628b3d53e8dd0c67 |
| rtl_trace.jsonl | 8a3b55efd690dcd6f939a1426275447bd8bf7bb7a470ff7c8761c0c6558ad5de |
