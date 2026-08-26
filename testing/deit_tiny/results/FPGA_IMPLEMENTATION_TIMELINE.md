# DeiT-Tiny FPGA implementation timeline

This timeline records the transition from the earlier CNN accelerator to a
timing-clean, device-resident DeiT-Tiny implementation on the PYNQ-Z2. It is
intended to preserve both the engineering sequence and the measured evidence
behind each architectural decision.

## Scope and measurement convention

- Target model: six-class fruit-quality DeiT-Tiny.
- Deployment precision: W4A8 for matrix weights and activations, with W8
  HeatViT selector weights in the balanced pruning profile.
- Board clock: 100 MHz unless a checkpoint explicitly states otherwise.
- Deployment contract: load the model once into the 64 MiB DDR arena, then use
  one host input write, one resident graph command, and one logits read for
  each image.
- PL windows are taken from device cycle counters. Host time additionally
  includes command transport and the input/output transfers.
- Results from different precision or token-count configurations are useful as
  implementation milestones, but only paired measurements within the same
  report should be interpreted as controlled speedups.

## Chronological implementation record

### 2026-08-22: architecture reset and first complete DeiT path

**`962b52d` — Replace CNN accelerator with DeiT transformer architecture**

- Removed CNN-specific operators and the previous 16x16 matrix architecture.
- Introduced one 8x8 DSP-based INT8 matrix core with a shared W8A8/W4A8 data
  path. Packed W4 weights are sign-extended only at array ingress.
- Added packed-weight readers, activation/weight buffers, LayerNorm, GELU,
  Softmax, residual arithmetic, cycle counters, and a Transformer-oriented
  command contract.
- Established 100 MHz place-and-route as a requirement rather than relying on
  simulation-only throughput estimates.

**`3d91886` — Complete streamed DeiT image inference runner**

- Connected preprocessing, patch embedding, class and position tokens, all 12
  encoder blocks, final normalization, and the six-class head.
- Produced the first complete physical image classification, but repeatedly
  uploading and reconstructing stages cost approximately 45.31 seconds.

**`66f7da1` — Keep DeiT model constants resident in DDR**

- Kept weights and normalization constants on the accelerator between runs.
- Reduced warm inference to a 29.33-second mean while preserving the output.
- Demonstrated that host-device traffic, rather than model arithmetic alone,
  was the dominant limitation of the streamed design.

### 2026-08-23: resident execution, instrumentation, and parallel GEMM

**`4703c9e` — Execute the full DeiT model as one resident graph**

- Expanded the allocator to a 64 MiB arena and retained constants,
  activations, workspaces, and descriptors in DDR3.
- Reduced the complete device graph to approximately 8.93 seconds and removed
  intermediate host round trips.

**`0f4661f` — Add precise DeiT FPGA cycle diagnostics**

- Added 64-bit operation, busy, memory-stall, input-stall, output-stall, and
  GEMM phase counters.
- Made per-module cycle reporting mandatory for subsequent performance tests.
- Established that the array's arithmetic peak was not the useful performance
  predictor: DDR scheduling and materialized intermediate tensors dominated.

**`d32237f` — Optimize DeiT DDR tile scheduling**

- Improved tile issue and DDR sequencing around the original 8x8 core.
- The measured one-core resident PL window reached 8.388 seconds.

**`882b7d0` — Add dual-core DeiT GEMM scheduling**

- Added a second independent 8x8 core and allowed the scheduler to pair M
  tiles while sharing B-tile delivery.
- Reduced the PL window from 8.388 to 4.517 seconds and GEMM from 5.200 to
  1.887 seconds, a 1.857x end-to-end speedup.
- Confirmed that another general matrix core would provide diminishing returns
  until data movement was reduced.

**`5a1678a`, `def7074` — Add and correct the shared rowwise engine**

- Added a dedicated reciprocal-based LayerNorm/Softmax engine.
- The first complete run exposed a one-column gamma/beta BRAM prefetch error
  that standalone tests had hidden. Prefix-graph captures localized the first
  mismatch to LayerNorm; distinct per-column test parameters reproduced it.
- Correcting parameter prefetch restored byte-exact LayerNorm and reduced the
  PL window to 3.143 seconds. LayerNorm fell by 73.37% and Softmax by 42.87%
  relative to the dual-core checkpoint.

**`9480a20` — Report module cycles for every DeiT FPGA run**

- Standardized saved JSON reports with per-operation, per-stage, per-layer,
  and GEMM role/phase data.

### 2026-08-24: W4A8 caching and elimination of intermediate passes

**`4e81378` — Optimize W4A8 DeiT FPGA execution**

- Changed the deployment path from W8 weights to packed W4 weights with INT8
  activations.
- Added a scheduler-local BRAM cache so each static W4 tile is loaded once per
  GEMM launch and reused across M-tile pairs. B loading fell by 84.4%, and the
  complete PL window fell from 3.576 to 2.729 seconds.
- Fused GEMM bias and requantization into tiled INT8 writeback, contracting the
  graph from 666 to 546 layers and reducing the PL window to 2.176 seconds.
- Replaced serialized requantization/writeback with a registered DSP pipeline
  and an 8x8 output tile buffer. GEMM writeback fell by 43.55%, bringing the
  PL window to 2.050 seconds.
- Extended producer-side layout generation until all standalone requantization
  and activation-B packing layers were removed. The graph contracted to 328
  layers, resident allocation fell to 14.72 MB, and the PL window reached
  1.162 seconds.
- Added wide W4 cache reads and overlapped writeback with later tile work. A
  read/write arbitration race initially dropped the second word of some rows;
  a collision regression reproduced the problem and writer priority fixed it.
  The corrected PL result was 1.096 seconds with exact logits.

### 2026-08-25: HeatViT, bandwidth reduction, and timing-closed sub-500 ms run

**`7d854f6` — Optimize W4 Load-B and validate HeatViT at 100 MHz**

- Integrated the toggleable balanced HeatViT profile, reducing tokens
  `197 -> 168 -> 133 -> 80` when its selector model is present.
- Added Load-B request pipelining, next-N-tile prefetch, W4 cache reuse, and
  BRAM bias caching.
- Reduced balanced HeatViT to 72,260,821 cycles, or 722.608 ms.

**`cb22354` — Accelerate GEMM writeback and rowwise DDR bursts**

- Added a second registered requantization lane to accept two GEMM accumulator
  outputs per cycle.
- Buffered eight LayerNorm/Softmax rows and emitted contiguous row-tile bursts.
- Reduced HeatViT GEMM writeback by 21.67%, Softmax by 41.75%, and the full PL
  window to 59,194,965 cycles, or 591.950 ms.

**`888064f` — Optimize resident model protocol latency**

- Reused the model's resident connection and removed avoidable command setup
  from repeated production inference.
- The exact PL window was 59,093,368 cycles; host-to-host latency was
  approximately 614.6 ms over repeated production runs.

**`cd7ef8f` — Optimize fused DeiT transformer execution**

- Replaced serialized exact LayerNorm/Softmax output arithmetic with a shared,
  seven-stage valid pipeline. Added sample/parameter prefetch to isolate BRAM
  lookup from numerator arithmetic.
- Fused the first MLP GEMM's requantized output directly through GELU, removing
  all 12 standalone GELU graph layers and approximately 54.6 ms.
- Implemented optional residual-to-LayerNorm fusion. An initial compiler pass
  incorrectly deferred residual tensors that later encoder blocks still used
  as skip inputs, producing incorrect logits. A liveness check now enables the
  fusion only when the residual output has no later consumer, restoring exact
  model output.
- Packed token embedding and concatenation writes in firmware.
- The first route of the deeper exact rowwise pipeline had WNS -1.691 ns.
  Registering the final sample/parameter fetch boundary improved the completed
  implementation to WNS +0.006 ns and WHS +0.033 ns at 100 MHz.
- The final balanced HeatViT run averages 416.63 ms for the resident device
  graph and 440.27 ms host-to-host. It is the first timing-qualified result
  below the 500 ms inference target and preserves the expected
  `apple_fresh` logits `[1, -3, 1, 1, 0, -6]`.

## Performance progression

| Checkpoint | Configuration | Measured time |
|---|---|---:|
| Streamed complete model | W8A8, repeated host staging | 45.31 s host |
| Resident constants | W8A8, staged graphs | 29.33 s host |
| One resident graph | W8A8, one core | 8.93 s PL |
| Optimized one-core scheduler | W8A8 | 8.388 s PL |
| Dual 8x8 cores | W8A8 | 4.517 s PL |
| Corrected shared rowwise engine | W8A8 | 3.143 s PL |
| Packed W4 BRAM cache | W4A8 | 2.729 s PL |
| Partial GEMM/requant fusion | W4A8 | 2.176 s PL |
| Pipelined GEMM writeback | W4A8 | 2.050 s PL |
| Full layout/operator fusion | W4A8 | 1.162 s PL |
| Wide-cache and write overlap | W4A8 | 1.096 s PL |
| Load-B prefetch | Balanced HeatViT W4A8 | 722.608 ms PL |
| Dual writeback and row bursts | Balanced HeatViT W4A8 | 591.950 ms PL |
| Persistent production transport | Balanced HeatViT W4A8 | 590.934 ms PL |
| Shared pipelined rowwise + fused GELU | Balanced HeatViT W4A8 | **416.63 ms PL average / 440.27 ms host average** |

The endpoints represent approximately a 102.9x reduction from the original
45.31-second streamed host path to the final 440.27-ms host path. This ratio is
an implementation-journey summary rather than a controlled single-variable
comparison because the later checkpoints also use W4A8 and token pruning.

## Final timing, area, and cycle distribution

| Implementation metric | Result |
|---|---:|
| Clock | 100 MHz |
| Setup slack | +0.006 ns |
| Hold slack | +0.033 ns |
| Slice LUTs | 26,695 / 53,200 (50.18%) |
| Slice registers | 30,740 / 106,400 (28.89%) |
| BRAM tiles | 42 / 140 (30.00%) |
| DSPs | 165 / 220 (75.00%) |

| Final module | Cycles | Time | Graph share |
|---|---:|---:|---:|
| GEMM | 25,931,472 | 259.31 ms | 62.21% |
| Softmax | 4,207,461 | 42.07 ms | 10.09% |
| Residual add | 3,632,378 | 36.32 ms | 8.71% |
| LayerNorm | 3,297,462 | 32.97 ms | 7.91% |
| Token concatenation | 2,947,054 | 29.47 ms | 7.07% |
| Token embedding | 1,098,517 | 10.99 ms | 2.64% |
| Token pruning | 572,584 | 5.73 ms | 1.37% |

GEMM remains the next optimization boundary. Its final phase counters are
7,641,192 Load-B cycles, 7,585,688 compute cycles, 7,371,938 write cycles, and
4,694,036 Load-A cycles. These phases overlap under dual-core scheduling and
must not be added as sequential latency. The W4 cache records 24,791 hits and
146 misses, a 99.4% tile hit rate, so future GEMM work should concentrate on
dynamic-B delivery, write overlap, and core utilization rather than enlarging
the static-weight cache.

## Verification record

- Correct physical classification and exact reference logits.
- 90 host tests passed; one optional test skipped.
- 65 firmware tests passed.
- All 27 RTL golden-trace simulations passed, including exact shared-rowwise
  equivalence and the integrated Transformer DDR top.
- Vivado implementation completed with zero failed routes and closed setup and
  hold timing at 100 MHz.

The final machine-readable capture is
[`fpga_cycle_report_100mhz_fused_shared_correct.json`](fpga_cycle_report_100mhz_fused_shared_correct.json).
Earlier controlled comparisons are preserved in the other Markdown and JSON
reports in this directory.
