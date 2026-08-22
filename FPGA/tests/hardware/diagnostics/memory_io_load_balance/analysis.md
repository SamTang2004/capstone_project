# Memory IO Load-Balance Stress

## Workload

The saturation test issues 1,024 sequential one-byte reads against a zero-stall
memory model. Sixteen 64-byte cache-line fills produce 256 AXI read beats. The
remaining requests hit in the on-chip read cache. The follow-up configurations
issue line-bounded 16-byte descriptors and optionally return four bytes per
cycle. Utilization curves use 64-cycle windows.

## Results

| Metric | Serialized baseline | Four-entry request-ahead | Change |
| --- | ---: | ---: | ---: |
| Sampled cycles | 5,393 | 4,372 | -18.93% |
| AXI read payload occupancy | 4.75% | 5.86% | +1.11 points |
| Output response occupancy | 18.99% | 23.42% | +4.43 points |
| Accepted request occupancy | 18.99% | 23.42% | +4.43 points |
| Correct output bytes | 1,024 | 1,024 | unchanged |

### Sixteen-byte descriptor follow-up

The same 1,024 bytes were then issued as 64 line-bounded, 16-byte requests.
Physical traffic remained 16 commands and 256 AXI payload beats.

| Metric | Queued scalar | Vector16 | Change |
| --- | ---: | ---: | ---: |
| Sampled cycles | 4,372 | 1,366 | -68.76% |
| AXI read payload occupancy | 5.86% | 18.74% | +12.89 points |
| Output response occupancy | 23.42% | 74.96% | +51.54 points |
| Requests | 1,024 | 64 | -93.75% |

Vector requests make 93.84% of response bytes consecutive. However, line fills
still begin exactly every 85 cycles: 16 cycles to fill a line, 64 cycles to
drain it through the 8-bit response port, and approximately five control cycles.
The driver cannot fill one cache bank while delivering another.

### 32-bit response follow-up

The response channel was widened to 32 bits with a four-bit byte-valid mask and
backpressure. The request queue was increased to eight entries and a second
line-fill slot was added. All configurations still transfer exactly 1,024
bytes using 16 memory commands and 256 physical AXI payload beats.

| Metric | Vector16 byte response | Wide32 response | Change |
| --- | ---: | ---: | ---: |
| Sampled cycles | 1,366 | 598 | -56.22% |
| AXI read payload occupancy | 18.74% | 42.81% | +24.07 points |
| Output response cycle occupancy | 74.96% | 42.81% | -32.15 points |
| Output payload | 1,024 bytes | 1,024 bytes | unchanged |
| Peak response width | 1 byte/cycle | 4 bytes/cycle | 4x |

The lower wide-response cycle occupancy is expected: each accepted response
cycle now carries four bytes. Effective delivered bandwidth rises despite the
smaller percentage of response-active cycles. The full transfer is 2.28 times
faster than the byte-response Vector16 run and 7.31 times faster than the
queued scalar run.

The request FIFO and registered lookup/prepared stages overlap cache-control
work and remove 1,021 scalar cycles under saturation. They do not increase
physical DDR traffic.

## Interpretation

This test establishes the benefit available to request-ahead-aware producers.
The deployable top currently leaves `READ_AHEAD_ENABLE` disabled because legacy
activation, bias, pooling, and tensor requesters permit only one logical request
in flight. Enabling it globally caused repeated lane requests and an orphaned
response in the activation write phase; the full-system test caught that fault.
Read-ahead is therefore explicit and opt-in until each producer either emits
multi-byte requests or tracks its outstanding requests.

The graph identifies two remaining bottlenecks. The wide interface removes the
8-bit response ceiling, but 16-byte requests drain in four cycles while a cold
64-byte line takes 16 AXI beats to fill. The speculative second line slot cannot
hide that miss unless descriptors are available far enough ahead.

The deployable top now carries the wide path directly from `memory_io_driver`
through Conv TileLoader into four independently addressed wave-memory banks.
The old byte adapter is retained only as a compatibility test and is absent
from the production source manifest. TileLoader also tracks eight accepted
response descriptors, permitting four requests to be outstanding before the
first response in the focused golden test.

The complete virtual top falls from 289,660 ns with the active byte adapter to
278,950 ns with direct four-lane writes and lookahead, a 10,710 ns or 3.70%
reduction. This is smaller than the isolated 7.31x memory-path result because
the full trace includes address generation, systolic execution, writeback, and
other operators that were not widened. The next optimization target is
multi-byte requests for activation, requantization, pooling, and tensor
operators, followed by a board diagnostics run to measure real HP0 stalls.

OOC synthesis with wide response, read-ahead, and an eight-entry queue meets a
150 MHz constraint with `+0.466 ns` setup slack. It uses 2,449 LUTs and 5,363
registers; the small request queues infer distributed RAM. The worst path is
the indexed line-cache selection into `read_word_data_r` and is 76.9% routing
delay.

The banked GEMM core OOC synthesis meets 100 MHz with `+3.103 ns` setup slack.
Explicit XPM storage maps to 16 RAMB36 blocks and zero LUTRAM; total core LUTs
fall from 17,373 to 15,895. The extra BRAM capacity is the physical cost of
supporting four independent byte addresses and a 128-bit wave read for both A
and B every cycle.

### All-consumer follow-up

The remaining CNN clients now issue word responses rather than passing through
the scalar compatibility path. Activation requests up to 16 bytes,
requantization requests up to four INT32 values, bias preload requests one
INT32 value, and both pooling engines request contiguous row runs up to four
bytes. Upsample and concat use the word protocol but still request one source
byte because their output-to-input reuse schedule has not yet been cached.

| Workload | Scalar/previous | Wide | Change |
| --- | ---: | ---: | ---: |
| Activation | 956 ns | 266 ns | -72.18% |
| Requantization | 845 ns | 516 ns | -38.93% |
| AveragePool golden set | 140,115 ns | 36,635 ns | -73.85% |
| Tensor-op golden set | 8,485 ns | 7,285 ns | -14.14% |
| Bias preload | 466 ns | 226 ns | -51.50% |
| Complete top versus direct-wide Conv | 278,950 ns | 261,640 ns | -6.21% |
| Complete top versus byte adapter | 289,660 ns | 261,640 ns | -9.67% |

The canonical full top passes its numerical golden trace. Its 100 MHz OOC
synthesis reports `+0.205 ns` WNS and zero failing endpoints after four timing
repairs: balanced registered MaxPool reduction, balanced burst-limit
calculation, a registered loader-to-BRAM payload, and an elastic shared-write
register. The last change also reduced the full simulation by 4,620 ns because
the writer can accept a new operator request while the previous request drains.

Production-top utilization is 30,162 LUTs, 27,130 registers, 16 RAMB36,
2 RAMB18, and all 220 DSP48 blocks. The current critical path is the Conv setup
calculation from `conv_kernel_area_r` to `conv_weight_tile_stride_r`, rather
than memory arbitration or pooling control.

Artifacts:

- `summary.json`: aggregate measurements
- `utilization_windows.csv`: 64-cycle utilization samples
- `utilization_curves.png`: presentation-ready raster graph
- `utilization_curves.svg`: scalable graph
- `baseline/memory_io_stress.csv`: serialized trace
- `queued_lookup_compatible/memory_io_stress.csv`: request-ahead trace
- `vector16/memory_io_stress.csv`: 16-byte descriptor trace
- `wide32_pingpong_q8_trace/memory_io_stress.csv`: corrected four-byte response trace
- `../../../../hardware/memory_io_driver_150mhz_timing.rpt`: wide-path OOC timing
- `../../../../hardware/memory_io_driver_150mhz_utilization.rpt`: wide-path OOC utilization
- `../../../../hardware/gemm_core_wide_timing.rpt`: banked-core 100 MHz timing
- `../../../../hardware/gemm_core_wide_utilization.rpt`: banked-core BRAM mapping
- `../../../../hardware/final_top_wide_timing.rpt`: canonical 100 MHz top timing
- `../../../../hardware/final_top_wide_utilization.rpt`: canonical top hierarchy
