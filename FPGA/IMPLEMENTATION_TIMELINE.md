# BudgetCUDA Implementation Timeline

This file is the chronological engineering record for BudgetCUDA. It tracks
what changed, why it changed, what was tested, and what was learned. The
architecture guide describes the current system; this timeline explains how
and why the project arrived there.

## Maintenance Rule

Update this file in the same commit whenever any of the following occurs:

- the host, protocol, firmware, RTL, register map, memory path, clock, or
  deployed bitstream architecture changes;
- synthesis or implementation is run for a changed design;
- a mass test is run, including a dataset evaluation, at least 100 repeated
  inferences, a broad RTL regression, or a board stress/performance run; or
- testing discovers a new correctness, timing, utilization, or deployment
  constraint, including failed and reverted experiments.

Every new entry must record:

1. date and short checkpoint name;
2. branch and commit, using `pending` until the commit exists;
3. architecture or implementation delta;
4. hypothesis being tested;
5. exact test conditions, including board, clock, model/data, warmup, and run
   count where applicable;
6. correctness, latency, cycle, timing, and resource results that were
   actually measured;
7. links to raw reports, traces, or test scripts;
8. discoveries, rejected assumptions, and the next action.

Do not silently replace an old result when conditions change. Add a new entry
and state why the measurements are or are not comparable.

## 2026-08-19: 100 MHz Latency Diagnosis

**Branch:** `DO-NOT-MERGE-Checkpoint-1`
**Commit:** pending
**Status:** repeated board diagnostics complete

### Conditions And Results

- Ran four instrumented CNN samples on the timing-closed 100 MHz PYNQ-Z2
  image. Mean summed layer time was 139.642 ms and cycle counts were stable.
- Ran 100 complete inferences with one persistent loaded model. All runs
  matched the software device and ONNX tolerance. Mean latency was 185.506 ms,
  median 187.812 ms, P95 195.613 ms, and maximum 200.222 ms.
- Conv consumed 77.780 ms, requantization 30.569 ms, pooling 20.633 ms, and
  activation 10.653 ms.
- Only 110,399 of 7,774,133 Conv cycles (1.4%) were compute-wait cycles. Output
  writing, loading, zeroing, and accumulation dominated the controller.
- Compared with the true-90.91 MHz JIT baseline, Conv cycles increased 6.5%
  and loader receive cycles increased 26.1%, despite identical cache hits,
  misses, fill cycles, and delivered bytes.

### Discovery

The cache timing pipeline is latency-pipelined but not throughput-pipelined:
`read_req_ready` blocks new requests during lookup and delivery. It pays the
new register stages serially for every read, largely cancelling the 10% clock
increase. Separate write, execute, and read TCP connections add approximately
30-45 ms between firmware time and application-visible latency.

### Artifacts And Next Action

- `tests/hardware/diagnostics/timing_closed_100mhz/analysis.md`
- `tests/hardware/diagnostics/timing_closed_100mhz/run_1/`
- `tests/hardware/diagnostics/timing_closed_100mhz/run_2/`
- `tests/hardware/diagnostics/timing_closed_100mhz/run_3/`

Next, add request/owner queues around the registered cache lookup so hits can
be accepted continuously, then fuse Conv writeback with requantization and
activation. Persistent host connections are a separate application-latency
optimization.

## 2026-08-19: Routed 100 MHz Timing Closure

**Branch:** `DO-NOT-MERGE-Checkpoint-1`
**Commit:** pending
**Status:** timing signoff, golden simulation, and board validation complete

### Change

- Split the DDR read path into request capture, registered associative-cache
  lookup, and result/command launch stages.
- Replaced the write-coalescer `base + count * 4` admission path with a
  registered next-address recurrence.
- Replaced the normal TileLoader row multiplication with an incrementing row
  base, removing a variable multiply from its steady-state request path.
- Made the production Vivado build run post-route physical optimization when
  required and reject release artifacts with negative setup or hold slack.
- Restored the firmware and systemd boot defaults from 90 MHz to 100 MHz only
  after the final routed image passed timing.

### Hypothesis

Registering the cache decision and converting repeated address products to
recurrences should remove control and address generation from the 100 MHz
critical path without changing DDR ordering or arithmetic results.

### Verification Conditions

- Target: PYNQ-Z2 `xc7z020clg400-1`
- Production constraint and deployed FCLK0: 100 MHz
- Virtual tests: focused DDR burst/cache test, TileLoader trace, and complete
  deployable-top golden simulation
- Board tests: upload/DRAM smoke, host-presliced Conv, Conv spatial ordering,
  complete 17-layer fruit CNN, and one instrumented CNN run
- 150 MHz status measured by applying a report-only 6.667 ns constraint to the
  final 100 MHz routed checkpoint

### Results

- Complete top simulation passed at 255,940 simulated ns. The updated memory
  test confirms that a miss command launches three cycles after read-request
  acceptance.
- The memory I/O driver closes 150 MHz out of context with WNS `+1.281 ns`.
- Final production route closes 100 MHz with WNS `+0.196 ns`, WHS `+0.021 ns`,
  zero failing endpoints, and zero DRC errors.
- Final utilization is 26,851 LUTs (50.47%), 27,111 registers (25.48%), five
  BRAM tiles (3.57%), and all 220 DSP48s.
- The next setup limiter is Conv command preparation from `a_stride` to the
  host-presliced weight-tile stride: 9.659 ns data delay, 15 logic levels.
- The unchanged routed design projects to WNS `-3.137 ns` at 150 MHz. Reaching
  150 MHz therefore requires another architectural pipeline pass; it is not a
  supported runtime clock for this bitstream.
- The timing-closed artifact programmed successfully at 100 MHz. Model upload,
  presliced Conv, and spatial-order tests passed. The complete CNN remained
  bit-exact with the software device and produced logits `[-25, 33, -20]`.
- One diagnostic run measured 139.753 ms firmware execution and 155.557 ms
  host-observed execution. This single run is not directly comparable to a
  mass latency test; it confirms correctness and identifies Conv layers 4, 0,
  and 8 as the largest remaining cycle consumers.

### Artifacts

- `hardware/pynq_z2_timing_impl.rpt`
- `hardware/pynq_z2_timing_150mhz_projection.rpt`
- `hardware/pynq_z2_utilization_impl.rpt`
- `hardware/memory_io_driver_150mhz_timing.rpt`
- `hardware/vivado/build_pynq_z2_bitstream.tcl`
- `hardware/vivado/report_pynq_z2_150mhz_projection.tcl`
- `tests/hardware/diagnostics/timing_closed_100mhz/`

### Next Action

Pipeline Conv command setup if 150 MHz is pursued, then rerun implementation
to reveal the next routed path. For inference latency, prioritize the Conv DDR
and array-duty bottlenecks shown by diagnostics rather than increasing FCLK0
beyond the signed-off 100 MHz value.

## 2026-08-19: JIT Conv Activation Address Generator

**Branch:** `DO-NOT-MERGE-Checkpoint-1`
**Commit:** pending
**Status:** RTL, golden simulation, and board validation complete at 90.91 MHz

### Change

- Split Conv activation address generation into a dedicated incremental
  descriptor frontend and the existing request/receive backend.
- Added a four-entry descriptor FIFO carrying DDR address, burst length, tile
  destination, padding, and final-descriptor metadata. Address preparation can
  now run while the previous request is being accepted or received.
- Replaced per-run Horner multiplication with nested coordinate counters and
  additive address recurrence. Tensor products are paid once during tile setup.
- Added a shared, DSP-free, registered 32x16 LUT multiplier for setup products.
  The steady-state descriptor path is split into coordinate check, burst
  preparation, emission, and advance stages to close at 100 MHz.

### Hypothesis

Generating activation burst descriptors ahead of the AXI consumer should hide
address-control latency behind DDR/cache traffic. Incremental recurrence should
remove the prior approximately 17 address/control clocks paid for each request
without taking DSP48s away from the fully allocated MAC array.

### Verification Conditions

- Target: PYNQ-Z2 `xc7z020clg400-1`
- Target clock: 100 MHz
- Focused traces: 3x3 input, 2x2 interior window; 3x3 padded window; repeated
  and offset activation tiles; ordinary and host-presliced weight tiles
- Full virtual path: canonical deployable PL top with GEMM, Conv2D,
  requantization, activation, pooling, upsample, and concat
- Synthesis: out-of-context post-synthesis timing and utilization for the new
  activation address generator

### Results

- Address generator golden trace passed contiguous and padded descriptors at a
  deterministic four-cycle descriptor interval.
- Integrated TileLoader golden trace passed all activation and weight bytes.
  The small eight-byte activation case completed in 100 clocks, equal to the
  prior implementation because its one-time setup dominates this tiny case;
  the FIFO hid the registered descriptor interval behind request/receive work.
- Complete deployable top golden simulation passed at 253,120 simulated ns.
- OOC timing at 100 MHz: WNS `+1.196 ns`, zero failing endpoints.
- OOC resources: 1,545 LUTs (2.90%), 1,052 registers (0.99%), zero BRAM, and
  zero DSP48s.
- Full board implementation used 26,998 LUTs (50.75%), 27,182 registers
  (25.55%), 5 BRAM tiles (3.57%), and all 220 DSP48s. Routed 100 MHz timing
  was WNS `-0.448 ns`, so the image was board-validated at the timing-safe
  90.91 MHz setting (`FCLK0_CTRL=0x00100b00`) instead of overclocking it.
- Changed the FPGA boot service to default `BCUDA_FCLK_MHZ` to 90 so a reboot
  cannot silently return this placement to 100 MHz. The environment value can
  be raised after a future image closes timing at that frequency.
- The upload/DRAM smoke, host-presliced Conv smoke, Conv spatial-order probe,
  GEMM row-order probe, and complete 17-layer CNN all passed on the PYNQ-Z2.
  The full CNN remained bit-exact with the software device and produced logits
  `[-25, 33, -20]`.
- Against the prior cache-8 result at the same true 90.91 MHz, aggregate Conv
  hardware cycles fell from 10,640,148 to 7,296,323 (`31.4%`). Loader address
  cycles fell from 3,901,943 to 1,655,560 (`57.6%`), while receive cycles were
  effectively unchanged (1,682,231 versus 1,681,553). This isolates the gain
  to control/address work rather than a change in DDR traffic.
- Firmware layer time fell from 174.126 ms to 137.339 ms (`21.1%`) and host
  execute round trip fell from 188.615 ms to 153.216 ms (`18.8%`) in the
  matched diagnostic runs.
- A balanced 300-image board run completed without a service or FPGA-manager
  fault: accuracy was 99.0%, ONNX agreement was 99.33%, mean latency was
  185.772 ms, p95 was 190.176 ms, and maximum latency was 193.447 ms. ONNX
  Runtime accuracy on the same images was 99.67%; the two additional FPGA
  disagreements remain consistent with the deployed truncating fixed-point
  contract.

### Discoveries

- A latency-pipelined FSM was not a throughput pipeline: the old loader
  restarted approximately 17 control stages for every request.
- Direct synthesis of the setup arithmetic inferred 13 DSP48s, which cannot be
  integrated because the deployed array already uses all 220 DSPs.
- Forcing those full multipliers into LUTs used zero DSPs but produced WNS
  `-8.343 ns`. Registering descriptor validation, burst selection, and address
  advance reduced the final OOC path to 8.797 ns.
- The canonical all-test `xvlog` invocation remained CPU-active without output
  for 30 minutes and was stopped. The focused benches and production-top-only
  regression compiled and passed normally, isolating that delay to the
  monolithic regression flow rather than the new RTL.
- The JIT generator removed most address-control work, but DDR receive time,
  output writing, requantization, pooling, and low array utilization remain.
  The next optimization should use the new breakdown rather than revisiting
  address recurrence. A separate placement/timing pass is required before
  returning the service clock to 100 MHz.

### Artifacts

- [JIT address generator RTL](hardware/rtl/conv2d_activation_address_gen.sv)
- [DSP-free setup multiplier](hardware/rtl/lut_mul32x16_pipeline.sv)
- [Address generator golden bench](hardware/sim/tb_conv2d_activation_address_gen.sv)
- [Integrated TileLoader bench](hardware/sim/tb_conv2d_tile_loader_ddr.sv)
- [OOC synthesis script](hardware/vivado/synth_conv2d_activation_address_gen_ooc.tcl)
- [OOC utilization report](hardware/conv2d_activation_address_gen_utilization.rpt)
- [OOC timing report](hardware/conv2d_activation_address_gen_timing.rpt)
- [Board diagnostic report](tests/hardware/diagnostics/jit_conv_addr_90mhz/fruit_cnn_diagnostic.md)
- [Matched-clock comparison](tests/hardware/diagnostics/jit_conv_addr_90mhz/comparison.md)

## 2026-08-19: Host-Presliced Conv Weights and Measured DDR Path

**Branch:** `DO-NOT-MERGE-Checkpoint-1`  
**Commit:** this checkpoint commit  
**Status:** deployed and board-validated

### Change

- Moved static Conv2D weight slicing from the PL TileLoader to the Python host
  compiler. OIHW INT8 weights are packed into 16x16 tile-major
  `B[K lane][N lane]` order before model upload.
- Added a versioned `CONV_PRESLICED_WEIGHTS` IL/firmware control flag and a
  matching PL address path. Dynamic activations remain NCHW and are still
  lowered into A tiles by the PL Conv loader.
- Replaced the simulated/general AXI bridge with the native 32-bit AXI3 HP0
  path used by the PYNQ-Z2 deployment.
- Fixed the deployable configuration at one 16x16 INT8 systolic array and
  removed obsolete production sources from the canonical build list.
- Expanded the shared read cache from one to eight 64-byte lines and added
  cycle-accurate Conv controller, loader, cache, and DDR counters through
  AXI-Lite, firmware diagnostics ABI v2, and the Python report exporter.
- Corrected the FCLK0 90 MHz setting from `0x00200b00` to `0x00100b00` and
  selected the timing-clean 100 MHz setting `0x00200500` for deployment.

### Hypothesis

Prepacking immutable model weights on the host should remove repeated OIHW
weight address generation from every inference. A small multi-line cache
should then retain the working DDR lines long enough to expose the remaining
Conv bottleneck.

### Verification Conditions

- Board: PYNQ-Z2, Zynq-7020
- Model: 17-layer signed INT8 fruit CNN
- Input seed: `2026`; one warmup run
- Compute: one 16x16 array
- Final PL clock: 100 MHz
- Read cache: eight 64-byte lines
- Virtual verification: complete top-level operator golden simulation,
  focused cache simulation, 72 host tests, and 50 firmware tests

### Results

- Output logits remained `[-25, 33, -20]` in the paired diagnostic runs.
- Firmware layer time: `158.508 ms`.
- Execute round trip: `172.733 ms`.
- Host total: `174.521 ms`.
- Conv2D total: `106.669 ms`.
- Routed timing: WNS `+0.127 ns`, TNS `0`, WHS `+0.030 ns`.
- Implemented resources: 25,287 LUTs (47.53%), 26,068 registers (24.50%),
  5 BRAM tiles (3.57%), and 220 DSPs (100%).
- Cache: 216,398 hits, 15,911 misses, 93.15% request hit rate, and about
  1.12x DDR read amplification.

Exact Conv controller attribution over 10,662,956 cycles:

| Section | Cycles | Conv share |
| --- | ---: | ---: |
| Load A | 4,924,576 | 46.184% |
| Write C | 2,386,564 | 22.382% |
| Accumulate | 1,066,496 | 10.002% |
| Load B | 910,111 | 8.535% |
| Zero A and B | 1,066,496 | 10.002% |
| Bias preload | 189,108 | 1.774% |
| Systolic compute wait | 110,399 | 1.035% |
| Other controller cycles | 9,206 | 0.086% |

### Discoveries

- The prior nominal 90 MHz deployment was actually running at 45.45 MHz.
  `0x00200b00` encoded divisors 2 and 11. This clock error was the largest
  cause of the apparent regression from the earlier 223.581 ms execute result.
- Host preslicing removes weight-format work, but it does not eliminate the PL
  loader: activation im2col and tile-cache filling remain on board.
- The cache expansion materially improved latency, but a 93.15% hit rate means
  cache capacity is no longer the primary limit.
- Systolic compute accounts for only 1.035% of Conv cycles. Activation address
  generation/A-tile construction is now the largest target, followed by
  writeback and serial accumulator movement.

### Artifacts

- [Host preslice golden generator](tests/hardware/create_conv_weight_preslice_golden_trace.py)
- [Preslice golden trace](tests/hardware/golden_traces/conv_weight_preslice_trace.json)
- [Final diagnostic analysis](tests/hardware/diagnostics/conv_profile_cache8_100mhz/analysis.md)
- [Final per-layer report](tests/hardware/diagnostics/conv_profile_cache8_100mhz/fruit_cnn_diagnostic.md)
- [Timing report](hardware/pynq_z2_timing_impl.rpt)
- [Implementation utilization](hardware/pynq_z2_utilization_impl.rpt)

### Next Action

Replace the sequential activation coordinate/address FSM with contiguous
window/run generation or a decoded activation-window cache. Then overlap A
loading, accumulation, and C writeback across adjacent tiles and re-run the
same cycle profile.

## 2026-07-27: TileLoader Control Pipeline

**Commit:** historical checkpoint; see repository history  
**Status:** accepted for timing closure

- Split burst-length calculation and final-beat advancement into registered
  states.
- Improved routed WNS from `-0.480 ns` to `+0.089 ns` at 100 MHz.
- Firmware time increased from `219.465 ms` to `224.595 ms`; the added boundary
  cycles traded latency for timing closure.
- Rejected direct final-beat advancement because it restored cycles but
  produced WNS `-0.075 ns`.
- Raw comparison: [tileloader_control_pipeline/comparison.md](tests/hardware/diagnostics/tileloader_control_pipeline/comparison.md).

## 2026-07-27: PE Operand Pipeline

**Commit:** historical checkpoint; see repository history  
**Status:** accepted

- Registered both PE operands before multiply and accumulation in LUT and DSP
  branches.
- Improved routed WNS from `-0.919 ns` to `-0.480 ns` and moved the worst path
  from the PE multiplier into Conv TileLoader control.
- Firmware layer time changed from `221.046 ms` to `219.465 ms`; logits stayed
  identical.
- Raw comparison: [pe_pipeline2/comparison.md](tests/hardware/diagnostics/pe_pipeline2/comparison.md).

## 2026-07-27: Bursty Conv2D Loader

**Commit:** historical checkpoint; see repository history  
**Status:** accepted as the INT8 performance baseline

- Replaced scalar Conv reads with line-bounded burst requests.
- Reduced firmware layer time from `502.753 ms` to `221.046 ms` and execute
  round trip from `514.968 ms` to `223.581 ms` at 100 MHz.
- Reduced combined Conv2D time from `445.275 ms` to `163.519 ms`.
- The provisional build still had routed WNS `-0.919 ns`; later PE and loader
  pipelines addressed timing.
- Raw comparison: [bursty_conv/comparison.md](tests/hardware/diagnostics/bursty_conv/comparison.md).

## 2026-08-19: Memory IO Request-Ahead Load Balancing

**Branch:** DO-NOT-MERGE-Checkpoint-1
**Commit:** pending
**Status:** simulated and OOC synthesized; board integration gated

### Change

- Added a four-entry read-request FIFO plus registered lookup and prepared
  response stages to `memory_io_driver`.
- Made request-ahead explicit with `READ_AHEAD_ENABLE`. Legacy top-level
  consumers remain in one-request mode until their response tracking is
  converted.
- Corrected activation and bias preload behavior when a response arrives during
  `READ_REQ`; the byte is now consumed rather than acknowledged and discarded.
- Added a 1,024-request saturation test, cycle CSV traces, and Python utilization
  plotting.

### Results

- Saturation cycles: `5,393 -> 4,372` (`18.93%` reduction).
- AXI read payload occupancy: `4.75% -> 5.86%`.
- Byte-response occupancy: `18.99% -> 23.42%`.
- A follow-up 16-byte descriptor run reduced cycles again to `1,366`, raised
  response occupancy to `74.96%`, and raised AXI payload occupancy to `18.74%`.
- All 22 active RTL golden testbenches pass, including the complete PL top.
- Memory-driver OOC timing at 150 MHz: `+0.989 ns` setup slack.
- OOC utilization: 1,786 LUTs, 5,300 registers, 0 DSPs, and 0 BRAM tiles;
  request descriptors infer small distributed RAMs.

### Discoveries

- A request FIFO cannot be enabled transparently for a producer that does not
  count outstanding requests. Doing so repeated activation lane requests and
  left an orphaned response blocking writeback; the full-system test caught it.
- Lookup overlap reduces control bubbles, but one-byte requesters still dominate
  the memory wall. Under ideal backing memory, physical read payload occupancy
  remains only 5.86%.
- Vector requests make 93.84% of response bytes consecutive, but the current
  8-bit response port caps throughput at 100 MB/s at 100 MHz. Two line banks and
  a 32-bit response stream are required to approach continuous HP0 bandwidth.

### Artifacts

- [Stress analysis](tests/hardware/diagnostics/memory_io_load_balance/analysis.md)
- [Utilization graph](tests/hardware/diagnostics/memory_io_load_balance/utilization_curves.png)
- [Plot generator](tests/hardware/plot_memory_io_utilization.py)
- [Saturation testbench](hardware/sim/tb_memory_io_load_balance_stress.sv)

### Next Action

Convert requantization to one four-byte request, activation to one 16-byte
block request, and pooling to bounded row runs. Then widen the internal response
stream to 32 bits and ping-pong two 64-byte line banks so line fill overlaps line
delivery. Add outstanding-response counts at the owner mux before enabling
request-ahead in the deployed top.

## 2026-08-19: 32-bit Memory Response Widening

**Branch:** DO-NOT-MERGE-Checkpoint-1
**Commit:** pending
**Status:** simulated, integrated, and OOC synthesized; board benchmark pending

### Change

- Added an optional 32-bit read-response channel with lane keep and final-word
  signaling to `memory_io_driver`.
- Preserved the byte response for bias, activation, requantization, pooling,
  and tensor operators.
- Added an eight-entry request mode queue, a second line-fill slot, and a
  four-word elastic adapter for the existing Conv TileLoader.
- Connected wide TileLoader requests in the deployable top and added explicit
  owner release after the adapter's final byte is accepted.
- Added wide-response saturation and adapter backpressure golden tests.

### Results

- Vector16 byte-response cycles: `1,366`.
- Wide32 cycles: `598`, a `56.22%` reduction and `2.28x` throughput increase.
- Wide32 versus queued scalar: `7.31x` throughput increase.
- AXI payload occupancy: `18.74% -> 42.81%`.
- Correct payload remained exactly 1,024 bytes, 16 commands, and 256 AXI beats.
- All 24 active RTL golden testbenches pass, including the full DDR top.
- Wide/read-ahead OOC timing meets 150 MHz with `+0.466 ns` setup slack.
- OOC utilization is 2,449 LUTs, 5,363 registers, 0 DSPs, and 0 BRAM tiles.

### Discoveries

- Four-byte delivery fixes the memory driver's response-width ceiling.
- The second line slot does not improve this 16-byte-descriptor test because a
  response drains in four cycles while a cold line needs 16 payload cycles.
- The deployable Conv path remains byte-wide at the wave-memory writer. The
  compatibility adapter prevents data loss but does not by itself reduce CNN
  cache-fill cycles.
- True end-to-end widening requires a banked four-lane wave-memory writer and
  lookahead ownership that permits another request while the current line is
  being consumed.

### Artifacts

- [Wide memory driver](hardware/rtl/memory_io_driver.sv)
- [Compatibility adapter](hardware/rtl/read_word_to_byte_adapter.sv)
- [Wide stress test](hardware/sim/tb_memory_io_load_balance_stress.sv)
- [Utilization analysis](tests/hardware/diagnostics/memory_io_load_balance/analysis.md)
- [Utilization graph](tests/hardware/diagnostics/memory_io_load_balance/utilization_curves.png)
- [150 MHz timing](hardware/memory_io_driver_150mhz_timing.rpt)

### Next Action

Replace the serial core load port with four independently banked write lanes.
Then let Conv TileLoader retain multiple outstanding descriptors so DDR fill,
adapter drain, and tile-cache writes overlap across adjacent runs. Measure the
change with a full CNN diagnostics run before attributing latency improvement.

## 2026-08-20: Direct Four-Lane Conv Tile Feed

**Branch:** DO-NOT-MERGE-Checkpoint-1
**Commit:** pending
**Status:** simulated and OOC synthesized; board benchmark pending

### Change

- Connected the memory driver's 32-bit response directly to Conv TileLoader.
- Added an eight-entry response-metadata queue so JIT address generation,
  request issue, and response retirement operate independently.
- Added four-lane tile writes with stride-one activation/presliced-weight loads
  and stride-16 raw OIHW weight loads.
- Replaced the serialized array operand memory with four modulo-wave banks per
  operand and explicit byte-write XPM block RAMs.
- Added per-owner outstanding-request tracking and enabled request-ahead only
  for the converted TileLoader path.
- Removed the byte adapter from the production source manifest while retaining
  its isolated compatibility test.

### Results

- Wide saturation: 1,024 bytes in `597` reported cycles versus `4,371` scalar
  cycles (`7.32x` throughput).
- Focused Conv TileLoader test proves four requests outstanding before the
  first response and verifies every four-byte bank write.
- Full virtual top: `289,660 ns -> 278,950 ns`, a `3.70%` reduction.
- All 25 active hardware golden testbenches pass.
- GEMM core OOC timing at 100 MHz: `+3.103 ns` WNS.
- GEMM core storage maps to 16 RAMB36, zero LUTRAM; total LUTs fall from
  17,373 to 15,895 (`1,478` fewer).

### Discoveries

- Widening the memory driver alone was insufficient because the byte adapter
  and scalar wave-memory write port serialized each response downstream.
- Explicit block-memory primitives are required for these shallow, wide banks;
  `ram_style="block"` still inferred 2,048 LUTRAMs.
- Four independently addressed bytes require physically banked memories. The
  resulting BRAM allocation trades unused capacity for the required write-port
  count and 128-bit compute read bandwidth.
- End-to-end latency improves less than the isolated memory path because legacy
  scalar operators and Conv address/writeback phases remain in the trace.

### Artifacts

- [Wide Conv loader test](hardware/sim/tb_conv2d_tile_loader_ddr_wide.sv)
- [Banked GEMM core](hardware/rtl/gemm_16x16_core.sv)
- [Memory load-balance analysis](tests/hardware/diagnostics/memory_io_load_balance/analysis.md)
- [GEMM core timing](hardware/gemm_core_wide_timing.rpt)
- [GEMM core utilization](hardware/gemm_core_wide_utilization.rpt)

### Next Action

Convert the remaining scalar DDR consumers to bounded word requests, then run
board diagnostics to separate HP0 service gaps from Conv address-generation,
compute, and writeback cycles.

## 2026-08-20: All-Wide CNN Memory Clients And 100 MHz Closure

**Branch:** DO-NOT-MERGE-Checkpoint-1
**Commit:** pending
**Status:** simulated, synthesized, implemented, and board validated

### Change

- Converted activation, requantization, bias preload, AveragePool, MaxPool,
  upsample, and concat to the 32-bit response interface.
- Batched activation reads by 16 INT8 values, requantization reads by four
  INT32 values, and pooling reads into bounded four-byte row runs.
- Added a balanced, registered MaxPool lane reduction and balanced Conv loader
  burst-limit calculation.
- Registered the loader-to-BRAM payload and CNN memory-client arbitration.
- Added a one-entry elastic shared-write register that sustains one request per
  cycle while breaking the address-coalescer ready loop.

### Results

| Workload | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Activation | 956 ns | 266 ns | 72.18% |
| Requantization | 845 ns | 516 ns | 38.93% |
| AveragePool golden set | 140,115 ns | 36,635 ns | 73.85% |
| Tensor-op golden set | 8,485 ns | 7,285 ns | 14.14% |
| Bias preload | 466 ns | 226 ns | 51.50% |

- Direct-wide full top: `278,950 ns -> 261,640 ns`, a further `6.21%`
  reduction. Relative to the byte-adapter top (`289,660 ns`), the total
  reduction is `9.67%`.
- The final golden top passes GEMM, Conv2D, requantization, activation,
  AveragePool, MaxPool, upsample, and concat.
- Canonical production-top OOC timing at 100 MHz: `+0.205 ns` WNS, zero TNS,
  and zero failing endpoints.
- OOC utilization: 30,162 LUTs, 27,130 registers, 16 RAMB36, 2 RAMB18, and
  220 DSP48 blocks.
- Fresh PYNQ-Z2 place-and-route at 100 MHz closes with `+0.176 ns` WNS,
  `+0.030 ns` WHS, zero failing endpoints, and no unrouted or partial nets.
- The deployed image reports FPGA manager `operating`, FCLK0 at exactly
  100 MHz, native 32-bit HP0, and both BudgetCUDA services active.
- All 17 CNN layer boundaries match the software device exactly. One full
  synthetic inference completed in `136.690 ms` after a `653.781 ms`
  one-time upload.
- A persistent 100-inference load test passed exact FPGA/software comparison
  on every run: mean `148.617 ms`, p50 `150.863 ms`, p95 `164.108 ms`, and
  maximum `175.227 ms`.
- The balanced 300-image fruit test reached `99.000%` board accuracy and
  `99.333%` ONNX Runtime agreement: apple `97/100`, banana `100/100`, and
  orange `100/100`. Mean board latency was `146.396 ms`, p95 `160.048 ms`,
  and maximum `172.727 ms`.
- Board diagnostics measured `104.018 ms` inside firmware. Conv2D layers
  consumed `65.34%` of that time and reported only `0.209%` to `0.595%`
  arithmetic utilization, so the remaining limit is dataflow scheduling and
  occupancy rather than functional correctness or timing closure.

### Discoveries

- Scalar CNN consumers remained visible in end-to-end latency after the Conv
  loader itself was widened; bounded operator-level descriptors remove that
  serialization.
- Serial MaxPool comparison produced the original `-6.666 ns` OOC WNS.
  Registering a balanced reduction moved the critical path out of pooling.
- A combinational write-ready loop through the coalescer coupled unrelated
  operator FSMs. The elastic write register both closes timing and improves
  simulated throughput by absorbing transient backpressure.
- The design now consumes all 220 PYNQ-Z2 DSP48 blocks. The next arithmetic
  feature cannot assume spare DSP capacity.
- PYNQ 3.1.1 does not expose the legacy `/dev/xlnk` device used by the old
  direct-CMA HP0 smoke script. Production validation therefore uses the
  BudgetCUDA PS service and its XRT-backed contiguous arena, which exercises
  the deployed host-to-PS-to-PL path.

### Artifacts

- [Canonical OOC synthesis](hardware/vivado/synth_final_top_wide_ooc.tcl)
- [Production-top timing](hardware/final_top_wide_timing.rpt)
- [Production-top utilization](hardware/final_top_wide_utilization.rpt)
- [Memory load-balance analysis](tests/hardware/diagnostics/memory_io_load_balance/analysis.md)

### Next Action

Use the board diagnostics to balance Conv2D address generation, HP0 service,
tile fill, compute, and writeback occupancy. The next data-path target is
contiguous batching for upsample and concat, which currently use the wide
channel with one-byte descriptors and remain address-generation dominated.

## 2026-08-20: Calculation Pipeline Zero-Fill and Retirement Removal

**Branch:** DO-NOT-MERGE-Checkpoint-1
**Commit:** pending
**Status:** simulated, synthesized, deployed, and board validated

### Change

- Replaced physical 256-entry A/B zero fills with one-cycle logical lane
  invalidation and validity masks at the GEMM core inputs.
- Reworked the accumulator as a pipelined read/modify/write engine with an
  explicit commit indication and one XPM simple-dual-port block RAM.
- Replaced per-output Conv2D writer address multiplication with setup-time
  plane/channel calculation and address recurrences.
- Added an eight-entry writer request FIFO so accumulator reads overlap HP0
  write backpressure and up to eight reads may be outstanding.
- Added stale-memory invalidation, sustained accumulator traffic, writer
  backpressure, and updated full-top golden checks.

### Hypothesis

The diagnostic split showed that computation itself consumed only 1.6% of
Conv2D controller cycles. Eliminating serialized zero fills, accumulator
read/modify/write bubbles, and writer address setup should reduce the dominant
calculation-side overhead without changing numerical results or the DDR
protocol.

### Verification Conditions

- Exact GEMM and CNN operator payload comparison in XSIM.
- A stale, nonzero BRAM payload must be masked after logical invalidation.
- The accumulator must sustain 768 updates and return all 256 golden values.
- Writer results must remain exact under changing AXI write backpressure.
- Accumulator storage must synthesize into block RAM and DSP usage must not
  exceed the device's existing 220 blocks.
- The complete PS/PL image must pass setup and hold signoff at 100 MHz.

### Results

- All 25 active hardware golden testbenches passed before the explicit XPM
  conversion; the focused accumulator and complete top passed again with the
  XPM implementation.
- Full-top virtual runtime fell from `261,640 ns` to `156,040 ns`, a `40.36%`
  reduction with identical GEMM, Conv2D, requantization, activation, pooling,
  upsample, and concat results.
- The accumulator maps to one `RAMB18E1`; the old 88-cell distributed-RAM
  implementation is gone.
- Production OOC synthesis passed at 100 MHz with `+0.205 ns` WNS, 31,371
  LUTs, 27,668 registers, 16 RAMB36, one RAMB18, and 220 DSP48 blocks.
- The complete PYNQ-Z2 image routed with no failed nets or DRC errors. A
  post-route AggressiveExplore pass closed timing at `+0.004 ns` WNS and
  `+0.027 ns` WHS. Final utilization is 31,258 LUTs, 28,281 registers, 16.5
  BRAM tiles, and 220 DSP48 blocks.
- A fresh bitstream, HWH, BIN, and XSA were generated and deployed. FPGA
  manager reports `operating`, and both BudgetCUDA services are active.
- Every boundary in the 17-layer CNN passed exact FPGA/software-device
  comparison. The first full synthetic inference fell from `136.690 ms` to
  `100.155 ms`, a `26.73%` reduction.
- Firmware diagnostics fell from `104.018 ms` to `67.316 ms`, a `35.28%`
  reduction. The three main Conv2D layers fell from 6,793,604 controller
  cycles to 3,129,037 cycles, a `53.94%` reduction.
- In those Conv2D layers, combined zero-fill fell to 4,160 cycles,
  accumulation fell to 536,640 cycles, and writer retirement fell to 276,384
  cycles. A/B loading is now dominant at 2,040,969 cycles; arithmetic compute
  remains 110,240 cycles.
- The persistent 100-inference test passed every exact comparison: mean
  `96.693 ms`, p50 `100.714 ms`, p95 `102.583 ms`, and maximum `105.260 ms`.
  Mean latency improved `34.94%` from the prior `148.617 ms` result.
- The balanced 300-image test preserved `99.000%` FPGA accuracy and `99.333%`
  ONNX Runtime agreement. Mean latency was `98.019 ms`, p95 `104.027 ms`, and
  maximum `113.180 ms`, improving mean latency `33.05%` from `146.396 ms`.

### Discoveries

- The prior accumulator inference directive was insufficient; explicit XPM
  instantiation is required to guarantee block-RAM mapping for this access
  pattern.
- The calculation controller can remove roughly 40% of virtual pipeline time
  without adding DSPs, confirming that retirement and housekeeping were major
  costs independent of DDR bandwidth.
- After optimization, OOC timing is led by Conv2D weight-stride setup. In the
  fully routed image, the final repaired path is in AveragePool address-plane
  logic. The systolic MAC path is not the 100 MHz timing limiter.
- Removing calculation housekeeping exposed A/B tile loading as the dominant
  Conv2D phase. Writer and accumulation are no longer the principal runtime
  blockers, while the compute phase itself is essentially unchanged.
- Initial TCP probes were blocked by the host execution sandbox even though
  ICMP was available. Direct FTDI D2XX UART access confirmed the board and all
  services were healthy before deployment resumed through approved SSH.

### Artifacts

- [Accumulator RTL](hardware/rtl/accumulator_buffer.sv)
- [Conv2D writer RTL](hardware/rtl/conv2d_tile_writer_ddr.sv)
- [Production top RTL](hardware/rtl/bcuda_tiled_systolic_ddr_top.sv)
- [Routed timing](hardware/pynq_z2_timing_impl.rpt)
- [Routed utilization](hardware/pynq_z2_utilization_impl.rpt)
- [PYNQ-Z2 bitstream](hardware/bcuda_pynq_z2.bit)

### Next Action

Optimize the remaining 2,040,969 A/B load cycles. Focus on sustained loader
request/receive overlap and cache-line reuse without regressing the exact
17-layer comparison, 99% balanced accuracy, or 100 MHz timing signoff.

## 2026-08-21: Native 64-bit A/B Loading and Write Adaptation

**Branch:** DO-NOT-MERGE-Checkpoint-1
**Commit:** pending
**Status:** simulated, synthesized, deployed, and board validated

### Change

- Widened HP0 and the shared DDR line-fill path to 64 bits. Each cache-line
  miss now returns eight native beats, and TileLoader exposes one eight-byte
  qword per accepted clock to the banked A/B wave memories.
- Expanded the read cache to sixteen 64-byte lines and retained queued
  lookahead descriptors so address generation can overlap active delivery.
- Replaced unsafe 32-bit narrow writes on HP0 with a native-width packer. It
  aligns `AWADDR`, recomputes physical `AWLEN`, supports odd logical counts and
  upper-lane starts, and emits byte strobes for partial first/final beats.

### Verification Conditions

- A/B qword responses must preserve all eight INT8 lanes under backpressure.
- Four logical writer words must become two ordered native HP0 beats.
- Odd and maximum-length bursts beginning at address `+4` must preserve every
  byte and produce the rounded-up physical beat count.
- The complete PYNQ-Z2 image must close setup and hold at 100 MHz.
- Board activation, GEMM, Conv2D, and full CNN comparisons must remain exact.

### Results

- Focused AXI3 bridge golden simulation passed native packing, odd counts,
  upper-lane starts, and 64-bit line reads.
- The integrated Conv2D writer-to-memory-to-AXI test passed with four logical
  words emitted as two native beats in 30 clocks.
- The complete image synthesized, placed, routed, and generated bit/HWH/BIN/XSA
  artifacts. Timing signoff passed at `+0.067 ns` WNS and `+0.016 ns` WHS.
- Deployed artifact hashes are `f0e36c1b...fca46548` for the bitstream and
  `1983f0bd...e948251` for the HWH.
- After a clean power cycle, FPGA manager reported `operating`, HP0 remained in
  native 64-bit mode, and both BudgetCUDA services became active.
- The activation lane probe returned all 32 bytes in order. The 4x4 identity
  GEMM returned values 1 through 16 exactly, and presliced Conv2D returned
  `[-4, -2, 2, 4]` exactly.
- All 17 CNN layer boundaries passed FPGA/software-device comparison. The first
  full run took `87.521 ms` on FPGA and produced logits `[-25, 33, -20]`.
- The persistent 100-inference test passed every comparison: mean `83.921 ms`,
  p50 `86.724 ms`, p95 `89.156 ms`, and maximum `105.641 ms`. Mean latency is
  `13.21%` lower than the prior `96.693 ms` checkpoint.
- The balanced 300-image test preserved `99.000%` FPGA accuracy and `99.333%`
  ONNX Runtime agreement. Mean latency was `83.363 ms`, p95 `90.167 ms`, and
  maximum `111.235 ms`, improving mean latency `14.95%` from `98.019 ms`.
- Firmware diagnostics measured `52.497 ms`, down `22.01%` from `67.316 ms`.
  Host execute and total times were `63.547 ms` and `87.258 ms` respectively.

### Discoveries

- The generated block design and HWH confirm that HP0 is physically 64 bits;
  full-width reads return all eight beats correctly.
- The first packed-write prototype retained the logical 32-bit `AWLEN` on the
  physical 64-bit channel, making HP0 expect twice the transmitted beats.
- Narrow 32-bit writes on the 64-bit HP0 port were not reliable on the board.
  Native beats with explicitly packed words are the deployable write contract.
- Conv2D remains the largest operator family at approximately `26.8 ms` across
  the three main layers. Requantization and pooling are now the next largest
  non-Conv costs in the measured execution profile.

### Artifacts

- [AXI3 bridge RTL](hardware/rtl/memory_io_axi3_bridge.sv)
- [AXI3 bridge golden test](hardware/sim/tb_memory_io_axi3_bridge_burst.sv)
- [Integrated writer golden test](hardware/sim/tb_conv_writer_axi_path.sv)
- [PYNQ-Z2 bitstream](hardware/bcuda_pynq_z2.bit)
- [Routed timing](hardware/pynq_z2_timing_impl.rpt)
- [Board diagnostic report](tests/hardware/diagnostics/fruit_cnn_diagnostic.md)

### Next Action

Use the new diagnostic baseline to reduce Conv2D A/B load stalls further, then
optimize requantization and pooling without regressing exact layer comparison,
99% balanced accuracy, or 100 MHz timing signoff.

## 2026-08-21: Deterministic Ordered Conv2D Loading

**Branch:** DO-NOT-MERGE-Checkpoint-1
**Commit:** pending
**Status:** simulated, timing-signed, deployed, and board validated

### Change

- Reduced the Conv2D activation address generator from four clocks per
  descriptor to two while preserving stable ready/valid behavior under
  backpressure.
- Allowed the TileLoader backend to complete one descriptor and dequeue the
  next in the same clock, and added ordered response metadata for multiple
  outstanding DDR reads.
- Reserved response-ring capacity when replacing a completing descriptor.
  This prevents zero-padding descriptors from overtaking the capacity check
  and corrupting the eight-entry response-order ring.
- Removed the AXI-ready-to-descriptor-merge combinational feedback path by
  merging only when more than one descriptor is already queued.

### Verification Conditions

- Descriptor output must remain stable while `ready` is low.
- Wide loading must sustain multiple outstanding reads without changing
  im2col byte order or padded values.
- The first production CNN Conv shape (`3x64x64`, `3x3`, pad 1, tile 16x16)
  must complete under variable memory stalls without queue overflow.
- The complete operator path and all active hardware tests must pass.
- The routed PYNQ-Z2 image must close setup and hold at 100 MHz.

### Results

- All 27 active RTL simulations passed. The new production-size stress test
  completed in 423 clocks with 63 DDR requests and three requests observed
  outstanding.
- The wide-loader test sustained four outstanding requests with a maximum
  issue gap of two clocks and preserved all eight qword lanes.
- The integrated top passed GEMM, production-size Conv2D loading,
  requantization, activation, pooling, upsample, and concat golden traces.
- Full-top OOC timing passed at `+0.205 ns` WNS. The telemetry-instrumented
  board image routed at 100 MHz with `+0.008 ns` WNS and `+0.017 ns` WHS.
- After a physical power cycle, the activation lane probe returned all 32 bytes
  exactly. Captured HP0 telemetry was `ARLEN=7`, `ARSIZE=3`, `RBEATS=8`, and
  `RRESP=OKAY`, proving that the native 64-bit eight-beat line fill completed.
- Protocol GEMM returned values 1 through 16 exactly and host-presliced Conv2D
  returned `[-4, -2, 2, 4]` exactly.
- Every boundary in the 17-layer CNN matched the software-device golden model.
  One instrumented run executed on the FPGA in `74.974 ms` and produced logits
  `[-25, 33, -20]`.
- The 100-inference exact-comparison test passed with mean `80.601 ms`, p50
  `85.210 ms`, p95 `87.093 ms`, and maximum `94.063 ms`. Against the native
  64-bit checkpoint, mean improved `3.96%` and p95 improved `2.31%`.
- The balanced 300-image test preserved `99.000%` FPGA accuracy and `99.333%`
  ONNX Runtime agreement. Mean latency was `81.536 ms`, p95 `87.955 ms`, and
  maximum `93.092 ms`; mean improved `2.19%` from `83.363 ms`.
- Firmware diagnostics measured `52.019 ms`, down `0.91%` from `52.497 ms`.
  Across the three main Conv layers, loader address cycles fell `4.71%`, Load A
  cycles fell `3.98%`, and total Conv cycles fell `1.79%`.

### Discoveries

- Small primitive tests did not exercise enough padded descriptors to expose
  response metadata overflow. The real first Conv layer did, deadlocking the
  loader before any output was produced.
- A producer-side capacity check must include descriptors that complete and
  replace one another in the same clock; FIFO occupancy alone was insufficient
  because padding descriptors create metadata without issuing DDR traffic.
- Address order is now deterministic through generation, request issue, and
  response retirement. It produced a repeatable but modest board throughput
  gain; descriptor scheduling was one contributor, not the entire memory wall.
- The earlier one-beat failure was stale PS/PL interface state after an FPGA
  manager reload. The same instrumented image completed an eight-beat HP0 burst
  after a true cold boot. Soft reload success must not be inferred from copied
  bitstream files or service state alone.
- Conv2D still consumes about half of firmware layer time, while requantization
  and pooling together consume about another 39%. Their scalar DDR processing
  rates now limit end-to-end improvement as much as descriptor generation.

### Artifacts

- [Address generator RTL](hardware/rtl/conv2d_activation_address_gen.sv)
- [Conv2D TileLoader RTL](hardware/rtl/conv2d_tile_loader_ddr.sv)
- [Production-size loader stress test](hardware/sim/tb_conv2d_tile_loader_ddr_stress.sv)
- [Integrated top test](hardware/sim/tb_bcuda_tiled_systolic_ddr_top.sv)
- [PL register dump](tests/hardware/pynq_pl_register_dump.py)
- [Ordered-loader board diagnostics](tests/hardware/diagnostics/ordered_wide_loader_100mhz/fruit_cnn_diagnostic.md)
- [Ordered-loader comparison](tests/hardware/diagnostics/ordered_wide_loader_100mhz/comparison.md)
- [Routed timing](hardware/pynq_z2_timing_impl.rpt)

### Next Action

Optimize the remaining Conv activation and weight delivery loops as a balanced
pipeline, then widen or fuse requantization, activation, and pooling. Keep the
current `80.601 ms` sustained host and `52.019 ms` firmware measurements as the
new baseline, and require exact 17-layer comparison plus 99% balanced accuracy
for every subsequent image.

## 2026-08-21: Post-Route Timing Hotspot Audit

**Branch:** DO-NOT-MERGE-Checkpoint-1
**Commit:** pending
**Status:** routed-checkpoint analysis complete

### Results

- Regenerated 100 unique-endpoint setup paths from the exact routed checkpoint
  used by the deployed image. Timing remains met at `+0.008 ns` WNS.
- AveragePool plane-address generation is the worst path: `9.842 ns` datapath,
  14 logic levels, and `+0.008 ns` slack. Logic contributes `4.231 ns` and
  routing contributes `5.611 ns`.
- TileLoader prepared-length calculation is second at `+0.057 ns`, with a
  `9.681 ns`, 15-level datapath.
- Systolic accumulator-enable distribution starts at `+0.091 ns`; its worst
  path is 95.3% routing rather than MAC arithmetic.
- DDR cache writes start at `+0.094 ns`; their worst path is 84.2% routing.

### Discovery

The limiting AveragePool cone is the generic `ni * channels` operation feeding
`addr_plane_r`. It is recomputed for every valid pooling sample and maps to a
LUT/carry multiplier. Converting plane addressing to an incrementally updated
base should improve both timing margin and operator latency.

### Artifacts

- [Timing hotspot script](hardware/vivado/report_pynq_z2_timing_hotspots.tcl)
- [Detailed timing paths](hardware/pynq_z2_timing_hotspots.rpt)
- [Machine-readable path table](hardware/pynq_z2_timing_hotspots.tsv)
- [Timing analysis](hardware/pynq_z2_timing_hotspots_analysis.md)

### Next Action

Replace AveragePool's per-sample plane multiplication with a registered,
incremental plane base, update its latency-aware testbench, and rerun active RTL
tests plus full implementation. Then address TileLoader prepared-length logic.

## 2026-08-21: Repeat Cycle Report and Utilization Limit

**Branch:** DO-NOT-MERGE-Checkpoint-1
**Commit:** pending
**Status:** measured and analyzed

### Results

- A second independent diagnostic run measured `52.020 ms` of firmware layer
  time, reproducing the first `52.019 ms` result within approximately 0.002 ms.
- Across Conv layers 0, 4, and 8, Load A consumed `43.90%` of cycles,
  accumulation `20.43%`, Load B `14.72%`, writes `10.52%`, bias `5.64%`, and
  systolic compute only `4.28%`.
- The loader cache hit rate was `92.10%`. All `15,264` misses completed exactly
  eight HP0 beats each, but useful delivery averaged only `0.808 bytes/cycle`.
- The immediate limiter is request/response serialization in the internal read
  cache: request admission is blocked while response valid is asserted, so
  short im2col descriptors repeatedly expose lookup and handoff bubbles.

### Artifacts

- [Repeat board diagnostics](tests/hardware/diagnostics/ordered_wide_loader_100mhz/run_2/fruit_cnn_diagnostic.md)
- [Cycle-limit analysis](tests/hardware/diagnostics/ordered_wide_loader_100mhz/cycle_limit_analysis.md)

### Next Action

Decouple cache request admission from response delivery and queue ordered
response metadata. Once that stream remains occupied, overlap next-tile A/B
loading with current-tile compute and accumulation through separate wave banks.

## Entry Template

```markdown
## YYYY-MM-DD: Checkpoint Name

**Branch:** branch-name
**Commit:** pending
**Status:** simulated | synthesized | deployed | rejected

### Change
### Hypothesis
### Verification Conditions
### Results
### Discoveries
### Artifacts
### Next Action
```
