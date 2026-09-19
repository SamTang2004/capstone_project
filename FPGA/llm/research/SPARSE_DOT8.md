# Eight-lane sparse dot-product prototype

## Result and research implication

The work finder and cross-packet buffer now drive a working arithmetic core.
All three tested front ends produce the same signed integer results. Packing
reduces issued multiplications and fills output beats, but the eight-position
input scanner limits elapsed throughput on the tested resident-vector workloads.
See [SPARSE_DOT8_RESULTS.md](SPARSE_DOT8_RESULTS.md) for measured cycle and area
tables generated from the RTL simulation and synthesis reports.

## Architecture

`rtl/llm_dot_product_lane.sv` instantiates one actual DSP48E1 per lane, configured
as a signed 8x8 multiplier with AREG/BREG=2 and MREG/PREG=1. There are exactly
eight lanes, with independently supplied aligned operand pairs. No off-diagonal
cross products are generated.

`rtl/llm_dot_core8.sv` supplies eight accumulator contexts. Six pipeline stages
perform four-stage multiplication, tagged balanced reduction, and INT32 accumulation.
Valid, mask, context slots and last flags follow the four-stage product pipeline.
The original result tables describe the earlier one-stage multiplier; the updated
timing/latency comparison is in [DSP_PIPELINE_RESULTS.md](DSP_PIPELINE_RESULTS.md).
Each product carries a three-bit context slot. For each context, a balanced tree
adds matching products, using 17-, 18- and 19-bit intermediate sums. The reduced
sums are registered before the accumulator update. Multiple lanes may target the
same context on every cycle without losing updates or serializing collisions.
The accumulator update is modulo 2^32. No product is requantized before summation.

The core accepts up to eight useful products per clock when they are available.
After accepting the final beat, it drains the arithmetic pipeline and captures
all declared job results. Results and tags remain stable until result_ready.
The input-last beat may be empty, so a wave of entirely zero jobs still finishes.
One command/wave is active at a time; there is no concurrent-wave execution.

`rtl/llm_sparse_dot8.sv` compares three elaboration-time choices on this core:

| MODE | Input path | Work issued |
| --- | --- | --- |
| 0 | Masked dense control | Every valid original position, including zero products |
| 1 | Eight-lane finder only | Survivors packed within each original packet |
| 2 | Finder plus survivor buffer | Survivors packed across packets and jobs |

All modes apply the same magnitude threshold to A. In mode 0, discarded A values
are replaced by zero before multiplication; in modes 1/2 they are omitted. Sparse
modes also omit pairs whose aligned B value is exactly zero. This preserves the
same thresholded answer for all modes and does not confuse a quality change with
a hardware speedup. The source still supplies both A and B for every position;
there is no implemented sparse index intersection or DDR gather engine.

## Command and stream contract

The command specifies an eight-bit job mask, eight 32-bit external tags, a 16-bit
wave tag and W4/W8 weight mode. Input packets identify their job by a three-bit
slot in this command's tag table. The wrapper refuses undeclared slots through
in_ready, except a pure zero-mask wave flush without a job-last flag. Inputs must
stay stable until accepted. Command buses are captured at acceptance.

W8 operands use eight signed bytes per beat. In W4 mode, each B lane uses the low
nibble of its byte; the wrapper sign-extends it to eight bits. Upper nibble bits
are ignored. This experimental byte-lane ABI differs from the production GEMM
weight-packing ABI. There is no quantization-scale processing in this core.

`in_job_last` marks the final input packet of each job; `in_wave_last` marks the
final packet of the wave. End a wave without waiting for its job fences. The
packer retains its eight-record dispatch-fence queue, consumed internally here.
The externally returned result is an arithmetic completion after pipeline drain,
not merely the packer's dispatch fence. Every declared zero-work job gets zero.
Result masks identify valid slots; all undeclared result values are zero.

The compute wrapper narrows the packer's payload to the eight-bit aligned B value
and its internal tag to a three-bit slot. Original source lanes are unneeded after
pair formation and can be optimized out. External tags stay in the command table.
Thus the area is not directly comparable to the earlier generic packer configured
with 32-bit payloads and 32-bit per-entry tags.

Reset cancels commands, queued entries, arithmetic pipeline state and pending
results. Upstream/downstream must invalidate outstanding work. The protocol
assumes well-formed job endings and matched operands, and provides no watchdog
for a missing wave end. More than eight simultaneous jobs need a larger context
table or separate waves; no general unbounded tag allocator is claimed.

## Measurement boundaries and controls

The testbench measures clock-edge intervals from accepted command to the first
sampled result_valid, and separately from first accepted input to that result.
Final output backpressure is reported separately. Input bubbles, original-vector
scanning, classification, compaction, buffer flush and arithmetic drain are in
the cycle comparison. Host preparation, DDR transfers, global index matching,
model execution and power are excluded. The test clock is 100 MHz for simulation;
that is not a measured physical clock or throughput claim.

Each front end receives the same original vectors, in consecutive eight-position
packets, with the same job boundaries and threshold. Random masks are reproducible
independent Fisher-Yates shuffles with fixed nonzero counts in each operand.
Dense mode is a strong low-overhead streaming control for this input contract;
all modes use eight DSPs, though synthesis may specialize logic differently.

The 0.1%-A / 99%-B fixture has eight length-1,000 dot products. Each A has exactly
one nonzero and each B has ten zeros. In this particular seeded fixture, all eight
A nonzeros intersect nonzero B entries. This is eight actual useful products;
it is not a claim that the expected joint density is exactly 0.1%.

Issue-beat counts include empty control beats. For example, when the last full
packet is already dispatched before an empty wave end arrives, that end occupies
one additional beat but enables no multiplier. Report useful occupancy both per
nonempty compute beat and over the complete clock interval; they differ sharply
at low density. CEM gating prevents product-register updates for invalid lanes,
but unregistered DSP input logic and other circuitry may still switch. Issued
MAC reduction is not a measurement of energy savings.

The earlier diagonal adapter supplies a functional illustration of discarded
cross products, not a same-resource timing baseline for this experiment. Its
K+4 execution window excludes compaction and transport and uses a different
operand layout. No speedup against that number is claimed here.

## Verification and reproduction

```powershell
& FPGA/llm/sim/run_sparse_dot.ps1
& C:/Xilinx/Vivado/2024.2/bin/vivado.bat -mode batch -source FPGA/llm/vivado/synth_sparse_dot.tcl -log FPGA/llm/build/sparse_dot/synth.log -journal FPGA/llm/build/sparse_dot/synth.jou
.\.venv\Scripts\python.exe FPGA/llm/research/report_sparse_dot.py
```

XSim uses the real DSP48E1 simulation library and waits for vendor global startup
reset before traffic. A scalar signed 64-bit reference computes each thresholded
dot product, then compares its low 32 bits with every declared result. Checks
include all contexts, repeated same-context collisions, mixed contexts, W4 sign
extension including -8 and ignored upper bits, zero partners/jobs, X=8 dropping
everything, source bubbles, held result backpressure, command-bus capture, reset
of an in-flight last packet and immediate reuse of the same core. A length-131,080
dot product of -128 by -128 deliberately crosses INT32_MAX and must yield
0x80020000. Synthetic fixtures do not establish language-model quality.

Synthesis is isolated from the board build, uses xc7z020clg400-1, and checks for
exactly eight DSP48E1 cells and no black boxes in every mode. The 10 ns clock and
0.300 ns uncertainty are loaded before synthesis. Reports remain post-synthesis
estimates with no physical clock-source location or external IO delays; routing,
board integration and board measurements remain outstanding.

## Next experiment

The wide-scanner follow-up is now measured in [WIDE_FINDER.md](WIDE_FINDER.md).
It demonstrates gains with resident wide words, lost when supply is capped at
eight pairs per cycle, and identifies one-bank-per-cycle drain as another limit.

Improve useful operand supply before adding more multipliers: compare a wider
scanner with an indexed/previously compressed input source, charging index reads,
gathers and metadata. Retain the three measured controls and the same numerical
goldens. A fair compressed-input comparison must give every capable baseline the
same compressed data, or explicitly charge the conversion cost. This experiment
has not demonstrated that packing itself is a novel algorithm or a latency win.
