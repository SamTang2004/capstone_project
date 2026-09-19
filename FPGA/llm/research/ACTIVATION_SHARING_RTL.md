# Activation sharing in Vivado

This is a separate RTL experiment on `xc7z020clg400-1` using Vivado 2024.2,
the real DSP48E1 simulation model, and the existing eight-context INT32 dot core.
It does not change the production FPGA project or program the board.

## Three architectures

| Variant | Finder | Execution | Descriptor storage |
| --- | --- | --- | --- |
| Tree cached | Existing pipelined 64-position compaction tree | Finish selection, then replay each activation across output rows | 4096 x 20-bit synchronous BRAM |
| Bitmap cached | New hierarchical bitmap selector | Same preparation/replay phases | Same BRAM |
| Bitmap streaming | Same bitmap selector | Broadcast survivors as selection proceeds | None; synthesis removes unused cache |

The bitmap finder classifies 64 signed INT8 values in parallel into a mask using
`abs(value) >> shift != 0`. It stores the values in their original positions,
selects the first nonempty eight-bit group, then the first set bit within that
group. Each accepted output clears one set bit. It returns the original value
and source index, preserving order. An empty frame still returns a control marker.
No population count or relocation of every surviving value is needed.

One survivor per cycle matches an eight-row broadcast: each lane multiplies the
same activation by a different row's weight. An eight-survivor compactor is useful
for other dataflows, but this particular broadcast path cannot consume eight new
activations per cycle. The tree comparison uses activation-only records (no weight
payload), so its area should not be compared directly with the earlier paired-input
14,121-LUT routed prototype.

Cached variants consume at most one descriptor per cycle from either finder.
The tree's eight-entry output chunks are serialized while held under backpressure.
A pipeline register separates descriptor selection from BRAM write setup; the
existing command-start cycle drains that write before replay reads it. BRAM output
uses an unreset synchronous read register, allowing actual block RAM inference.

Streaming merges selection with execution and can discard the cache for one tile.
It emits a final zero-mask marker if trailing empty frames follow the last useful
activation. That marker costs an interface transaction but reads no weights.

## Interface and scope

- `cmd_rows`: 1..8 output rows. `cmd_shift`: 0..15. Shift zero removes exact zeros;
  positive shifts are approximate pruning. Values are not shifted before multiplication.
- At most 4096 activation positions, in frames of 64. `in_mask` marks valid positions;
  `in_last` marks the vector's final frame. An empty vector uses one empty last frame.
- `cmd_passes=1` shares selection. For the repeated-finding ablation, the source
  resends the same vector eight times (`cmd_passes=8`) before the final computation.
  Other positive four-bit pass counts are representable but the sweep tests 1 and 8.
  This ablation isolates repeated scanning, not the best alternative accelerator.
- Weight requests contain an index, activation, output-row mask and last flag.
  Memory must return weights **in order**, preserving activation/mask/last sideband.
  Empty requests must return a control response. Both directions support stalls.
- The testbench uses eight output-interleaved weight banks: row `r` maps to bank `r`.
  One request reads one weight per active row. Responses are delayed two cycles and
  buffered in eight entries. Additional request stalls and result stalls are tested.
  The weight-memory model and its sideband queue are **outside the synthesized DUT**.
- Activation assembly is charged at 8 or 64 bytes/cycle, including the first frame.
  Acquisition of the next frame can overlap current-frame processing. These are
  local-supply models, not a DDR controller or bus implementation.
- All variants use the same INT32 arithmetic and weight requests. Zero weights
  still consume their lane/read; no weight-sparsity compaction is claimed here.
- Cached RTL currently replays only one eight-row tile per command. Streaming
  also covers one tile. Reusing a retained cache across additional output tiles
  needs a tile/replay command extension; the earlier Python 32/64-row results are
  not claimed as RTL results.

## How to interpret scanning efficiency

Let `K` be input length, `S` the number of survivors, and `B` activation bytes per
cycle. With one descriptor delivered per cycle, an optimistic preparation bound is
`max(ceil(K/B), S)`. The report divides this bound by measured scanning cycles.
This measures closeness to the input/output service limits, not useful-MAC density.
Empty-frame handling and finite buffering can make the true bound tighter.

For eight rows with nonzero weights, useful compute utilization is
`8*S / (8*total_cycles)`. At 8-byte supply, 10% activation density can supply only
about 0.8 new activations/cycle: its ideal utilization ceiling is about 80%.
At 0.1% density that ceiling is about 0.8%, even with a perfect finder. Efficient
scanning and low compute utilization can therefore occur together.

To improve extreme sparsity, the next useful change is producer-generated masks
or hierarchical occupancy metadata that lets memory skip empty blocks. Generating
those masks, threshold changes, metadata bandwidth and storage must be charged.
For more than eight output rows, stream the first tile while caching descriptors
for later tiles; that extension is not part of this prototype.

## Verification and reproduction

```powershell
& FPGA/llm/sim/run_activation_sharing.ps1
& FPGA/llm/sim/run_bitmap_finder.ps1
& 'C:/Xilinx/Vivado/2024.2/bin/vivado.bat' -mode batch -source FPGA/llm/vivado/synth_activation_sharing.tcl -log FPGA/llm/build/activation_sharing_synth.log -journal FPGA/llm/build/activation_sharing_synth.jou
& 'C:/Xilinx/Vivado/2024.2/bin/vivado.bat' -mode batch -source FPGA/llm/vivado/implement_shared_activation.tcl -log FPGA/llm/build/activation_sharing_impl.log -journal FPGA/llm/build/activation_sharing_impl.jou
python FPGA/llm/research/report_activation_sharing_rtl.py
```

Integration checks scalar results, exact indices/values/order, descriptor counts,
weight reads, final markers, stable requests/results under stalls, signed thresholds,
partial frames/row groups, empty vectors, reset during collection and reset with
memory/compute work in flight. The standalone bitmap test covers every byte at
every position for every threshold, every survivor count 0..64, every singleton
position, and output backpressure.

Synthesis compares all variants under the same 10 ns clock/0.300 ns uncertainty.
Unplaced synthesis setup/hold estimates are not timing closure and lack IO budgets.
The separate implementation flow qualifies **bitmap streaming only** between
explicit shared-clock launch/capture registers. Its 676 launch and 307 capture
registers are timing-fixture overhead. Only the outer fixture-port paths are false;
all DUT and register-to-register paths remain timed. The fixture is not a functional
ready/valid adapter. Physical timing does not qualify the external weight-memory
system, board IO or DDR. See the generated results for actual closure status.

[Measured results](ACTIVATION_SHARING_RTL_RESULTS.md)
