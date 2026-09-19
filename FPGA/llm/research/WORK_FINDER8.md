# Eight-lane work finder and local relocator

## Predicate

`rtl/llm_work_finder8.sv` accepts eight values and an input validity mask.
X (`in_shift`) is the number of low bits ignored: bit X itself is included.
For an unsigned value, keep iff any bit in [WIDTH-1:X] is set, equivalently
`value >= 2**X`. A reduction OR suffices; no population count is needed.
The shift is used only for classification. Retained values are not shifted.

The default SIGNED_MAGNITUDE=1 applies the predicate to the unsigned magnitude
of a two's-complement value. This retains -128 at X=7 for WIDTH=8, and drops
-1 at X=1. SIGNED_MAGNITUDE=0 applies the user's literal bit test to raw bits;
in that mode negative signed encodings can pass because their sign bit is set.
All shift codes >= WIDTH drop every entry. X=0 removes only numerical zeros;
X>0 also removes nonzero values and requires separate model-quality evaluation.
The predicate operates in integer quantization units. Different scales imply
different real-value thresholds. Asymmetric quantization would require zero-point
handling upstream; this module assumes zero represents numerical zero.

Example in signed magnitude mode, X=2:

```
input values:  [-1, 0, 3, -4, 8, 1, -128, 7]
input mask:    11111111
packed values:[-4, 8, -128, 7, 0, 0, 0, 0]
source lanes: [ 3, 4,    6, 7, 0, 0, 0, 0]
out_count:    4
out_mask:     00001111
out_keep_mask:11011000
```

## Interface and relocation

Lane zero occupies the least significant packed slice. Survivors retain their
relative input order and occupy output slots 0 through out_count-1. Each keeps
its original, unmodified value, PAYLOAD_WIDTH-bit payload, and 3-bit source lane.
Unused output slots are zero. out_keep_mask identifies surviving original lanes;
out_mask identifies occupied packed slots. Configure payload to carry a global
index, paired operand and/or output tag as needed by the upstream contract.

This is a single-vector classifier, not a dot-product index matcher. For a pair
of operands, carry the aligned companion in the payload and supply its validity
through in_mask (or build a second predicate and AND the original-lane masks).
Never independently compact A and B and zip the resulting values: their original
indices may differ. Thresholding an individual operand is not a bound on product
error or on total accumulated output error.

One registered ready/valid output holds data under backpressure. An empty output
register accepts an input even with out_ready low; a consumed packet can be
replaced on the same clock edge. Reset cancels the pending packet and clears
outputs. A packet with no survivors emits a valid count=0 completion so its
acceptance is observable. Packet-level job completion metadata is not implemented;
per-lane payload alone cannot identify a job when every lane is dropped.

## Scope and next stage

This prototype uses combinational classification and stable compaction into an
elastic output register. It has one-cycle registered latency and can accept one
eight-entry packet per cycle when downstream accepts outputs, as an RTL interface
property. The current implementation uses a four-bit prefix rank and fixed-slice
selectors, verified with the same exhaustive test after optimization. Combined
finder/buffer synthesis results are in WORK_PACKER8.md; physical frequency and
board throughput remain unmeasured. No new source was added to the production
board manifest.

Compaction in this module is within each packet only. The now-implemented
`llm_work_packer8` wrapper connects it to a bounded survivor buffer that combines
packets and job tags. See [WORK_PACKER8.md](WORK_PACKER8.md) for wave flushing,
empty-job dispatch fences, verification and remaining hardware work.
This module still scans eight supplied entries per accepted cycle; it does not
avoid reading a densely stored vector or demonstrate system-level sparse speedup.

## Verification

Run from repository root:

```powershell
& FPGA/llm/sim/run_work_finder.ps1
```

Vivado 2024.2 XSim passed 7,355 scoreboard cycles and 5,345 accepted packets,
checking raw-bit and signed-magnitude instances in parallel. The independent
reference uses integer magnitude and threshold comparison, rather than the RTL's
shift/reduction predicate. Coverage includes all 256 values at all eight lanes
for all 16 shift encodings, every input-mask pattern, empty/full packets, payload
and source-index preservation, consecutive replacement, random input bubbles,
downstream stalls and reset cancellation. Unaccepted inputs are also varied
adversarially to check that pending output data cannot be overwritten.
