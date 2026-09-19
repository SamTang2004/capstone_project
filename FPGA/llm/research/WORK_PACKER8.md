# Eight-lane packing across packets and jobs

## Implemented boundary

The arithmetic integration is now available in [SPARSE_DOT8.md](SPARSE_DOT8.md),
with dense/local/packed controls and real DSP48E1 simulation. This document records
the generic standalone packer and its original verification/resource configuration.

`rtl/llm_work_packer8.sv` connects the existing magnitude/raw-bit work finder
to `rtl/llm_survivor_buffer8.sv`. The finder retains its original predicate;
the new buffer packs survivors from multiple input packets and different jobs
into a single eight-entry output. Input order is preserved exactly.

Example: jobs A, B, C, D contribute 3, 1, 2, 2 survivors. One output contains
`[A0, A1, A2, B0, C0, C1, D0, D1]`, with each entry's value, opaque payload,
original source lane, and job tag. Data and all sidebands remain stable while
the corresponding ready/valid output is stalled.

The buffer declares 16 entry slots and reaches at most 15 occupied slots under
its admission policy. A separate eight-record FIFO tracks job dispatch fences.
The finder can hold one additional packet of up to eight survivors. The buffer
stops accepting when its offered output stalls; it does not consume all spare
storage opportunistically. This simple policy makes the output boundary stable.

## Stream contract

- Each input packet belongs to `in_job_tag`. `in_payload` can contain the matched
  second operand, original global K index, destination information, or a packed
  combination. The three-bit source lane alone is local to the original packet.
- `in_job_last` marks the final input packet for a job, including jobs with no
  survivors. Assert it exactly once per job and do not submit more work for that
  job afterward. Use distinct tags for concurrently outstanding jobs.
- `in_wave_last` ends a ready work wave and requests a partial flush. It may be
  attached to a data packet, or sent on an empty packet with `in_job_last=0`.
  `in_wave_tag` is captured on that accepted packet. An input mask of zero makes
  a pure flush packet through the integrated finder.
- Survivors can mix across job boundaries within a wave. They never mix across
  wave boundaries. The buffer blocks further input until the final output of
  that wave is accepted; the upstream finder may hold the next packet meanwhile.
- Output `out_mask` marks slots 0 through `out_count-1`; unused slots are zero.
  `out_wave_last` and `out_wave_tag` identify the final dispatch beat of a wave.
  A completely empty wave emits an explicit valid count=0 final beat. If all full
  beats were already dispatched before an empty end marker arrives, another empty
  final beat is necessary; a previously accepted beat cannot be marked retroactively.
- `done_valid/done_ready/done_job_tag` is an independent, ordered **dispatch fence**
  channel. A fence becomes valid only after all survivors preceding that job's end
  marker have been accepted by the output consumer. Empty jobs still produce a
  fence, conservatively ordered behind earlier survivors. This does not acknowledge
  completed multiplication or accumulation. The future MAC stage must track that.
- Upstream must send a wave end without waiting for unfinished jobs' dispatch
  fences; otherwise an unfinished partial packet can wait indefinitely by contract.
- Reset cancels buffered entries, captured wave boundaries, and pending fences in
  both modules. Upstream/downstream must discard outstanding work on reset.

The standalone survivor-buffer input count must be in 0..8, with packed entries
in the low slots. Other counts are refused through in_ready=0. The integrated
finder always generates legal counts. All job/wave sidebands are registered on
the same handshake as the corresponding finder packet, including empty packets.

## Forward progress and throughput

Normally, dispatch occurs when at least eight entries are buffered or a wave
end has arrived. There is one additional condition: if the job-fence FIFO is full
and its head still waits for data dispatch, emit the available partial packet.
This prevents a deadlock where one survivor followed by many empty jobs fills
the fence FIFO before the producer can send its final flush. Such an output is
partial but does not assert out_wave_last. Its metadata remains stable under
backpressure. The testbench explicitly exercises this case while stalled.

With dense input and ready consumers, the integrated pipeline accepts eight
entries each cycle, verified over 32 consecutive packets. Earliest dispatch is
two clock periods after input acceptance. A wave boundary introduces an admission
bubble in the buffer. Storage shifting, compaction muxes, and the combinational
ready path are prototype choices; timing optimization can change this organization.

Packing full output beats is not proof of sustained eight-MAC-per-cycle utilization.
With only eight dense entries scanned each cycle and retained fraction d, the
average supply is at most 8*d survivors/cycle before stalls. At very low density,
filling an output still takes many scan cycles. A compressed source, a wider work
finder, or appropriate indexed operand gathering is needed to raise supply.
No multiplication, accumulator routing, index intersection, DDR gather engine,
power measurement or full-model speedup is implemented here.

## Verification

```powershell
& FPGA/llm/sim/run_work_packer.ps1
```

Vivado 2024.2 XSim: PASS, 9,044 scoreboard cycles, 6,661 accepted packets,
10,608 dispatched survivors, 2,361 job fences, 289 completed waves. Counts include
traffic across reset epochs; some accepted work is deliberately canceled by reset.

The independent reference filters raw input using integer magnitude/threshold
comparisons, then tracks a flat logical entry stream and ordered job/wave
frontiers. It compares all output data, payloads, source lanes, tags, masks and
unused slots, and rejects lost/invented/reordered survivors, early fences, wave
crossing, buffer overflow, stalled-output changes, and bounded-test deadlock.

Directed coverage includes every pair of survivor counts 0..8, mixed-job filling,
empty waves/jobs, a late empty flush, dense sustained admission, pressure-driven
partial dispatch under stalls, simultaneous input/output, and reset with full or
partial pending output. Two hundred deterministic randomized waves exercise
multiple packets per job, all shift encodings, legal source backpressure, output
stalls and completion stalls. Observed coverage: 930 mixed-job full beats,
3 pressure partials, 20 empty flushes, 1,396 output-stall cycles, 2,230 fence-stall
cycles and 1,073 concurrent input/output handshakes.

## Standalone synthesis

Reproduction script: `vivado/synth_work_packer.tcl`, targeting xc7z020clg400-1
with default widths (8-bit values, 32-bit opaque payloads, 32-bit job tags and
16-bit wave tags). It writes reports/checkpoint under `build/work_packer_synth`.
This is isolated from the production board manifest and does not program a board.

Standalone synthesis completed successfully in Vivado 2024.2 with **4,809 LUTs,
1,926 registers, zero BRAM and zero DSP**, and no inferred latches or black boxes.
The finder uses a four-bit prefix rank and fixed-slice selection; the reservoir
uses fixed eight-position tail movement and fixed-destination append selectors.
These avoid wide variable-position writes while preserving the tested behavior.
The complete front end occupies approximately 9.04% of this device's LUTs; further
optimization should consider metadata widths, physical fanout and pipelining.

The timing report is an unplaced post-synthesis estimate, with a 10 ns clock and
0.300 ns uncertainty applied after synthesis. Internal setup WNS is +0.804 ns;
hold WHS is -0.035 ns (273 failing hold endpoints), so this is **not timing closure**.
There are no unconstrained internal endpoints, but input/output delay constraints
and a physical clock-source location are absent. No 100 MHz hardware-performance
claim follows from this check; place/route and integration constraints are required.
