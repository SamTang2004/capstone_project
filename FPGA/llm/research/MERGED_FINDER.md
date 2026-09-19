# Cross-bank survivor merging

`llm_wide_work_finder` and `llm_wide_sparse_dot8` now expose `MERGE_BANKS` (default 0). Setting it to 1 merges partial banks within a captured frame. The default retains the original bank-dispatch baseline; production board sources are unchanged.

The eight-entry finders still perform signed-magnitude thresholding and stable local compaction. A prefix sum of their survivor counts gives each local survivor a global rank. Eight fixed output lanes select ranks `offset` through `offset+7`, retaining paired payloads and original source indices. After an accepted nonfinal chunk, the offset advances by eight. All local banks remain captured until the last chunk is accepted; a new frame may replace them on that same edge. Backpressure freezes the frame and offset. Reset discards both.

For example, eight banks containing one survivor each now produce one full output beat instead of eight single-survivor beats. A frame with 13 survivors produces beats of eight and five, regardless of bank boundaries. An empty frame still produces one zero-count final control beat. Every nonfinal output beat is full. Metadata is repeated per chunk, and job/wave end flags take effect only on the final chunk. The existing downstream buffer combines tails across frames and jobs; the unchanged tagged core performs arithmetic and reports actual completion.

This implementation uses combinational prefix ranks and fixed-slice selection, with no added pipeline register. Its cost is a wider selection network and a count-to-dispatch timing path. All survivors remain in the local finder registers, so it does not add a second full-frame data store. Counts need `clog2(SCAN_LANES+1)` bits to represent an entirely dense frame. Legal scan widths remain 8, 16, 32 and 64.

For a frame with `S` survivors in `B` nonempty banks, the old finder needs `max(1,B)` accepted output cycles; the merged finder needs `max(1,ceil(S/8))`. These counts exclude downstream stalls and final compute drain. At random 50% density and width 64, nearly every bank is nonempty, while the frame contains about 32 survivors. Merging therefore approaches four drain cycles instead of eight. Random frame tails raise the actual mean above four; the downstream tail buffer can fill compute beats but cannot recover a finder cycle already spent emitting a frame tail. At extremely low density, both designs usually emit one chunk per frame, so merging has little benefit.

The experiment preserves the previous fixtures and two supply models. Resident mode offers a complete wide word immediately. Capped mode charges acquisition at eight pairs per cycle. At width 64, byte-lane A/B inputs require 128 bytes per offered frame; this is a local supply assumption, not an implemented memory system. End-to-end utilization includes scan and drain cycles. Frame tails and acquisition still prevent universal full utilization.

## Reproduce

From the repository root in PowerShell:

```powershell
& FPGA/llm/sim/run_wide_finder.ps1 -MergeBanks
& FPGA/llm/sim/run_wide_sparse_dot.ps1 -MergeBanks
& FPGA/llm/sim/run_wide_finder.ps1
& FPGA/llm/sim/run_wide_sparse_dot.ps1
& C:/Xilinx/Vivado/2024.2/bin/vivado.bat -mode batch -source FPGA/llm/vivado/synth_wide_sparse_dot.tcl -log FPGA/llm/build/merged_synth.log -journal FPGA/llm/build/merged_synth.jou -tclargs merge
.\.venv\Scripts\python.exe FPGA/llm/research/report_merged_finder.py
```

Merged build outputs use separate `build/merged_*` folders. The report verifies conservation of fixture work and source-supply charges, identical scan-8/dense controls, and exact reproduction of the default baseline metrics. Synthesis asserts eight DSPs and no black boxes for each width.

See [measured comparison](MERGED_FINDER_RESULTS.md). Further design choices should use area and timing as well as cycle counts: pipelining the merger or staging smaller merges may be needed before physical implementation. General survivor compaction is not itself a novelty claim.
