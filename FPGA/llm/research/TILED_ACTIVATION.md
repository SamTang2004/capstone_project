# Producer summaries and multi-tile activation reuse

This experiment implements producer summaries and cross-tile reuse. The current
revision pipelines the shared bitmap finder and producer for 200 MHz, and specializes
the tiled core to fixed lane-to-row assignments. Earlier result tables are historical.

## Dataflow

`llm_activation_block_map` accepts eight signed INT8 activations per beat and
generates one occupancy bit per 64-position block. A bit is set when at least one
masked-in value satisfies `abs(x) >> shift != 0`. Up to 4096 positions require just
64 bits / eight bytes of metadata (0.195% of an INT8 vector's size).

`llm_tiled_activation8` accepts that map and a row count of 1..64. Its activation
memory request generator traverses either every block or just the set bits. Returned
frames retain their original block indices, so gaps do not renumber activations or
weight addresses. The existing bitmap finder selects survivors within fetched blocks.

Tile 0 broadcasts each survivor across up to eight rows while writing the same
`{12-bit index, 8-bit activation}` descriptor into a synchronous BRAM cache. After
the result is accepted, the controller starts the next eight-row tile and replays
the descriptors. The cache has 4096 entries (10 KiB logical capacity), mapping to
two RAMB36 plus one RAMB18. A partial final tile masks inactive lanes. Empty vectors
or all-empty block maps still generate a completion marker and zero-valued results.

Two runtime controls provide ablations on the **same hardware**:

| Skip empty blocks | Reuse cache | Behavior |
| --- | --- | --- |
| Off | Off | Scan the raw activation vector again for every output tile |
| On | Off | Fetch only occupied blocks, but repeat those reads for each tile |
| Off | On | Scan raw inputs once, stream tile 0, replay subsequent tiles |
| On | On | Fetch occupied blocks once, stream tile 0, replay subsequent tiles |

Unlike the previous artificial repeated-finding experiment, each tile here computes
different output rows in **every** mode. The dense case therefore stays compute-bound;
eliminating repeated scans need not reduce latency when scans already overlap compute.

## Metadata is not free

The testbench obtains the map from the actual RTL producer and checks it against an
independent scalar summary. For K=4096, the producer takes 515 measured cycles (command,
512 eight-byte beats and two pipeline-drain cycles) to make the map available. Output-stall testing is excluded
from this producer latency. The producer hardware cost is reported separately.

Two accounting cases are reported:

- **Resident metadata:** an upstream producer has already generated the summary while
  writing activations. Consumer latency still includes a one-cycle/eight-byte metadata
  read. The tap supports eight bytes/cycle; it cannot absorb a wider producer for free.
- **Separate metadata pass:** start with a stored raw vector, read it once through the
  metadata producer, then run the consumer. Add all producer cycles and K activation
  read bytes to the skip-enabled case. The producer also writes an eight-byte summary.
  These are sequential phases, with no assumed overlap.

If the map has to be built by a new standalone scan for just this invocation, charge
that preparation explicitly against reuse alone. The initial raw scan already
finds the exact selections needed for all later tiles. A summary can be generated
as part of existing production or reused across multiple consumers to avoid that cost.

The map is threshold-specific. The source vector, shift and map must agree and remain
unchanged for a command. A new vector or threshold needs a new summary. Every new
command invalidates the descriptor count; no stale cache contents are reused.

## Memory and timing contract

- At most 4096 activation positions; `cmd_blocks=ceil(K/64)`, in 0..64.
- `cmd_rows` is 1..64, `cmd_shift` is 0..15. Shift zero removes exact zeros; positive
  shifts change the computed function and require a separate accuracy evaluation.
- Activation requests contain a block index and last flag. Responses are ordered
  and preserve those fields, with a 64-byte frame and a validity mask for partial blocks.
- The test memory has eight finite frame slots. A single local read port assembles
  frames at eight or 64 bytes/cycle, plus two cycles response latency. Backpressure
  fills those slots and stops requests. Partial-block reads charge only valid bytes.
- Weight requests carry tile, index, activation, row mask and last. The model has
  eight output-interleaved banks and two-cycle ordered responses with an eight-entry
  queue. Every active lane reads one weight; zero weights are not skipped by an oracle.
- The 200 MHz full-rate experiment preserves bytes/cycle, doubling bytes/second
  relative to the 100 MHz baseline. A separate `MemoryScale=2` run doubles activation
  assembly/response times and weight service intervals/response latency in cycles,
  preserving baseline bytes/second for the consumer memory interfaces. Its metadata
  producer input remains full-rate, so only resident-map consumer times are compared.
- Memory interfaces must preserve sideband under stalls and return an empty control
  request without reading weights. No cross-tile memory responses may be reordered.
- Results are emitted one tile at a time and can stall. No next tile starts until the
  previous tile's result is accepted. Tile startup/drain overhead is included.
- Reset cancels the command and cache validity. Associated memory adapters must
  discard pre-reset responses as well; the testbench flushes both memory queues.
- Source-read metrics are local memory transactions, not DDR bursts or total system
  energy. Descriptor read/write bits are reported separately; reducing activation-source
  reads does not imply the same reduction in all on-chip memory traffic.
- Physical activation/weight memories, response queues, producer-to-consumer storage,
  DMA, DDR and board IO are outside the synthesized/routed units. Timing fixtures
  qualify each block separately, not an integrated memory subsystem.

## Verification

The shared arithmetic core now enables AREG/BREG=2 and MREG/PREG=1 in each DSP48E1.
Products and their valid/mask/slot/last metadata have four stages, followed by
reduction and accumulation. Throughput remains one eight-lane beat per cycle;
each tile drains three extra cycles relative to the original multiplier pipeline.
See [DSP pipeline experiment](DSP_PIPELINE_RESULTS.md) for timing and latency costs.
The current tiled core sets `FIXED_LANE_SLOTS=1`: lane i always accumulates row i.
Generic tagged cores default to arbitrary slots and retain the balanced reduction.
The new finder splits magnitude, threshold, group occupancy, group selection and
registered survivor output. Groups cost a preparation cycle between local streams.
The block requester similarly separates group and lane selection and registers
its output. No two tiles execute concurrently. See [200 MHz results](PIPELINE_200_RESULTS.md).

The matrix-vector scoreboard checks every row against signed scalar arithmetic, exact
source indices, tile order, masks, completion, activation reads, weight reads, cache
writes/reads, stable stalled requests/results, partial blocks and partial output tiles.
Tests cover random and clustered masks, all 16 shifts, W4-range values including -8,
empty vectors, all-empty maps, a full 4096-entry cache, and cancellation during first
tile selection, later-tile replay and producer generation. One correct tile completed
before a replay reset is included in the reported verified-tile/dot totals.

Producer testing separately covers all 256 byte patterns at every one of eight lanes
for all 16 shifts: 32,768 singleton cases. Singletons ensure another valid value in
the same block cannot hide a mistaken classification. Metadata is also checked while
the output is stalled and after resetting a partially generated nonempty map.

## Reproduction

```powershell
& FPGA/llm/sim/run_tiled_activation.ps1 -ClockMHz 200 -RunName tiled_activation_200
& FPGA/llm/sim/run_tiled_activation.ps1 -ClockMHz 200 -MemoryScale 2 -RunName tiled_activation_200_fixed_bw
& FPGA/llm/sim/run_bitmap_finder.ps1
& 'C:/Xilinx/Vivado/2024.2/bin/vivado.bat' -mode batch -source FPGA/llm/vivado/implement_tiled_activation.tcl -log FPGA/llm/build/pipeline200_impl.log -journal FPGA/llm/build/pipeline200_impl.jou -tclargs 5.000 tiled_activation_200_impl
& 'C:/Xilinx/Vivado/2024.2/bin/vivado.bat' -mode batch -source FPGA/llm/vivado/implement_block_map.tcl -log FPGA/llm/build/pipeline200_producer.log -journal FPGA/llm/build/pipeline200_producer.jou -tclargs 5.000 block_map_200
python FPGA/llm/research/report_tiled_activation.py
python FPGA/llm/research/report_pipeline_200.py
```

The implementation commands target `xc7z020clg400-1`, 5.000 ns and 0.300 ns clock
uncertainty, with explicit shared-clock boundary registers. Only paths outside the
launch/capture registers are false. No DUT or register-to-register paths are excepted.
The consumer fixture has 1072 registers; the producer fixture has 148. These are
excluded from DUT resource counts and functional latency, but included in placement.

[Measured results](TILED_ACTIVATION_RESULTS.md)
