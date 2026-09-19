# Wide work-finder experiment

## Scope

The new `llm_wide_work_finder` scans 8, 16, 32 or 64 candidate positions in
parallel. `llm_wide_sparse_dot8` connects it to the same survivor buffer and
unchanged eight-DSP arithmetic core used in SPARSE_DOT8. This experiment changes
the input organization, not the number of multipliers or accumulator contexts.
Measured results are in [WIDE_FINDER_RESULTS.md](WIDE_FINDER_RESULTS.md).

## Banked organization

Each bank uses the verified eight-entry magnitude classifier/compactor. All banks
capture a frame on one input handshake. They retain their aligned A/B entries,
source indices and frame metadata in registers. An arbiter selects the lowest
nonempty bank and emits its compacted entries, at most eight per cycle. Empty
banks are skipped combinationally, without a cycle per empty bank. Survivors
retain original order and frame-relative source indices (three to six bits).

The last chunk can be consumed while a replacement frame is accepted. The ready
contract ensures every local bank can accept that replacement. If all banks are
empty, one count=0 final chunk preserves the frame's metadata. Output metadata
repeats on every chunk; job/wave end flags take effect only on out_frame_last.
The existing buffer combines partial chunks across banks, frames and jobs,
and sends up to eight matched products to the core.

This is a banked finder, not a fully compacted 64-entry crossbar. It emits from
one nonempty bank per cycle. A half-full bank consumes an arbiter cycle even if
other banks also have partial work. Consequently a random 50%-dense workload
often requires one cycle per original eight-position bank regardless of scan
width. Increasing width mainly helps when many banks are completely empty.
The output buffer can make full compute beats, but cannot recover the bank-drain
cycles already spent collecting their entries.

For independently random retained density p, an eight-position bank is nonempty
with probability q = 1 - (1-p)^8. Skipping empty banks alone therefore has an
approximate cycle-reduction ceiling of 1/q, before scan-width and pipeline costs.
At p=50%, q is 255/256, so almost no banks can be skipped. At p=10%, q is about
0.570, predicting a ceiling around 1.76x even with very wide input. This explains
the moderate-density saturation in the measured banked design. Fixed-count
shuffled fixtures approximate, rather than exactly implement, independent masks.

The finder has one frame of storage, with replacement on its final drain cycle.
It does not independently queue or interleave multiple captured frames. A dense
64-position frame takes eight chunks to drain; the eight-DSP core, not the scan
width, bounds dense throughput. Job slots, final arithmetic results, W4 byte-lane
sign extension, INT32 wrap and reset behavior follow SPARSE_DOT8.md. Frame input
must stay stable until accepted. Only the listed scan widths are supported.

## Supply controls

All configurations receive the same original vectors, values, retained thresholds
and job ordering. The baseline is the existing MODE=0 dense eight-DSP core.
The new scan-8 configuration is also checked against the earlier packed path's
recorded cycle counts on the matching fixtures. Command-to-result intervals include
front-end admission, scan, drain, packing, flush and arithmetic pipeline latency.
Output stalls are measured separately.

Two distinct input supply contracts are simulated:

1. **Resident wide input:** The next complete frame is offered immediately.
   SCAN_LANES pairs can be supplied per accepted frame, subject to finder
   backpressure. This assumes banked/wide local storage can deliver that word;
   it does not include initial memory loading, prove BRAM banking, or measure DDR.
2. **Eight-pair supply:** Acquiring a frame costs ceil(valid positions / 8) cycles,
   with a minimum one cycle for an empty metadata frame. The first frame pays its
   acquisition cost. Acquisition of the next frame overlaps draining the captured
   frame; source stalls continue if the finder has not finished. Only actual
   positions are charged in partial tails. This is a testbench rate model, not an
   implemented DMA/prefetch unit; it does not claim optimal buffering under a
   bursty real memory system.

Every mode still reads every original A/B position. At width 64 the current
byte-per-operand ABI has a 1,024-bit (128-byte) data input, versus 128 bits at
width 8. At a hypothetical 100 MHz, offering a frame every cycle would require
12.8 GB/s versus 1.6 GB/s of local operand supply. These are interface arithmetic,
not measured bandwidth or achieved clocks. W4 still occupies a byte lane in this
prototype; packed external weight storage is not implemented here.

The W4 stall fixture inserts bubbles per submitted frame, so wider configurations
receive fewer injected bubbles. It is a correctness/backpressure stress case;
its cycle counts must not be used as a controlled width-speedup comparison.
It is excluded from the main performance tables, but retained in the raw CSV.

## Verification

The standalone 64-position finder reference checks every byte value at every
position for all 16 shift encodings, each isolated source index, random masks,
paired payloads, frame end markers, stall stability, atomic bank acceptance,
replacement and reset cancellation. It checks the original six-bit indices even
though the arithmetic wrapper no longer needs indices after operand pairing.

The compute test instantiates all four widths and the dense control, using the
actual DSP48E1 simulation library. Scalar signed dot-product goldens cover dense,
50%, 10%, 0.1% x 99%, all-zero, short/empty jobs, W4 thresholding, complete threshold
rejection, INT32 overflow, stalls, reset and immediate command reuse. Useful-product
counts and original source-position counts must match across configurations.
The tests wait for the vendor global startup reset before issuing traffic.

```powershell
& FPGA/llm/sim/run_wide_finder.ps1
& FPGA/llm/sim/run_wide_sparse_dot.ps1
& C:/Xilinx/Vivado/2024.2/bin/vivado.bat -mode batch -source FPGA/llm/vivado/synth_wide_sparse_dot.tcl -log FPGA/llm/build/wide_sparse_dot/synth.log -journal FPGA/llm/build/wide_sparse_dot/synth.jou
.\.venv\Scripts\python.exe FPGA/llm/research/report_wide_finder.py
```

Synthesis targets xc7z020clg400-1 with a 10 ns clock and 0.300 ns uncertainty,
checks exactly eight DSPs and no black boxes, and writes separate results for each
width. Reports are unplaced estimates. External IO delays and physical clock
location are unspecified, and no timing closure or board throughput is established.
The production board manifest and bitstream are not changed by these experiments.

## Interpretation

The wide finder can remove the scan bottleneck for very sparse resident inputs,
but cannot manufacture operand bandwidth. It also does not guarantee near-dense
useful utilization: in the extreme fixture only eight products exist among 8,000
positions, and scanning even 64 positions at a time takes many clocks.

The next architectural question is whether cheaply combining partial survivors
from several banks before arbitration can improve moderate random sparsity without
losing the area/timing advantage of the banked design. That mechanism would need
its own measured comparison; this experiment does not implement it.
