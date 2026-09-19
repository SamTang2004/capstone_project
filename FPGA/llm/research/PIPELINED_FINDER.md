# Pipelined cross-bank merging

Set `PIPELINE_MERGE=1` on `llm_wide_work_finder` or `llm_wide_sparse_dot8`. This selects `llm_pipelined_work_finder`; `MERGE_BANKS=0/1` with pipelining disabled retains the previous banked/combinational baselines. Production manifests and board bitstreams are unchanged.

## Data and flow control

An input FIFO holds two candidate frames. Its ready signal depends only on registered occupancy, cutting the backwards ready chain. It adds one cycle and sustains one frame per cycle when simultaneous enqueue/dequeue keeps it at one entry. A completely full FIFO must first pop before admitting another frame; tests cover this recovery. The eight-entry local finders still classify signed magnitudes and compact each bank stably. Their output records carry the original value, paired payload and full source index. At width 64, three registered tree levels combine compacted groups of 8 into 16, then 32, then 64. Each merge appends its right child at the left child's survivor count. Unused records are zero, allowing a shift and OR to form the stable concatenation. Record width is padded to a power of two so the shift selects whole records; constant padding is optimized away.

Each tree level has a valid bit and holds its data, counts and metadata under backpressure. Different frames may occupy different levels. The final level holds a complete compacted frame while a serializer advances in eight-record increments. A separate elastic output register holds each dispatched chunk until accepted. Empty frames still emit a zero-count final control beat; nonfinal chunks are full. A final-frame transfer can coincide with replacement by the next frame. Reset cancels all frame stages and the output register.

Vivado infers distributed RAM for operand storage in the input FIFO. Metadata uses a separate two-entry register array with `ram_style="registers"`. The initial all-RAM version met setup but retained a −0.057 ns hold path directly from RAM output to a metadata flip-flop even after hold-fix optimization. Separating metadata changes physical storage, not FIFO behavior or cycle count. The failed registered attempt is preserved locally in `build/pipelined_registered_ram_metadata/scan64`.

At scan width 64 this adds five registered stages along the forward path relative to the combinational merger: the input FIFO, three tree levels and the serializer output. It adds five cycles to the unstalled main fixtures, rather than five cycles per input frame, because adjacent frames overlap. Scan widths 8/16/32 add two/three/four cycles respectively. The final serializer still needs `max(1,ceil(survivors/8))` accepted cycles per frame; the pipeline does not remove frame-tail or input-supply limits. Full-pipeline backpressure is tested explicitly.

```mermaid
flowchart LR
    A[64 candidate pairs] --> Q[Two-frame input FIFO]
    Q --> B[Eight local finders: registered]
    B --> C[Four 16-entry merges: registered]
    C --> D[Two 32-entry merges: registered]
    D --> E[One 64-entry merge: registered]
    E --> F[Eight-entry serializer: registered output]
    F --> G[Existing survivor buffer]
    G --> H[Unchanged eight-DSP tagged core]
```

## Timing experiment

`implement_registered_sparse_dot.tcl` synthesizes, optimizes, places and routes the complete 64-position wrapper including survivor buffering and the eight-DSP core inside `llm_sparse_dot_timing_boundary`. The harness places explicit launch/capture registers around every DUT port on the same clock. These registers are preserved, including command, reset, ready/valid, operand and debug signals. It is a timing fixture, not a functional ready/valid transport wrapper; functional traffic is independently checked by the DUT scoreboard.

The target is `xc7z020clg400-1` at 10 ns, with 0.300 ns clock uncertainty and clock origin `BUFGCTRL_X0Y0`. Two false-path constraints exclude only fixture-input-to-launch-register and capture-register-to-fixture-output wiring. No DUT path or path between the launch/capture boundaries is excluded; no multicycle exception is used. The script reports setup, hold, route status, exceptions, DRC and utilization, and fails if worst setup or hold slack is negative. The banked control uses identical registered boundaries. Reported harness resources must be separated from DUT resources.

If an initial route misses setup or hold, `finish_registered_impl.tcl` preserves that checkpoint and runs post-route physical optimization followed by routing again. A hold failure selects `ExploreWithAggressiveHoldFix`; a setup-only failure selects `AggressiveExplore`. `refine_registered_sparse_dot.tcl` can resume this step from a completed checkpoint without repeating synthesis. These physical changes do not add RTL pipeline cycles. See [AMD's physical-optimization command reference](https://docs.amd.com/r/2024.2-English/ug835-vivado-tcl-commands/phys_opt_design).

The earlier external-port experiment (`implement_pipelined_sparse_dot.tcl`) used 2 ns maximum / 0.5 ns minimum IO delays relative to an ideal external clock. Before the input FIFO was added, the pipelined run failed with setup −1.002 ns (survivor-buffer state to `in_ready`) and hold −1.220 ns (external command tag to core register). The banked control failed at −1.105 / −1.129 ns. These failed reports remain in `build/{pipelined,banked}_sparse_dot_impl`. The final registered experiment is a different boundary model, not closure of that earlier external-port contract. We do not claim that contract now passes.

Shared-clock source/capture flops model the intended on-chip producer and consumer directly, without inventing external clock latency. AMD documents that clock latency must represent the actual external/OOC context and can be difficult to specify reliably: [Vivado 2024.2 clock-latency guidance](https://docs.amd.com/r/2024.2-English/ug949-vivado-design-methodology/Clock-Latency-at-the-Source).

Meeting the registered-boundary constraints establishes a routed block timing result for that environment. It does not establish board operation, DDR supply, pin timing or measured maximum frequency. Wider operand availability remains an assumption in the resident-source experiment. The timing harness's outer fixture interface is not timing-qualified and is not part of the measured DUT cycle interval.

## Reproduce

From the repository root in PowerShell:

```powershell
& FPGA/llm/sim/run_wide_finder.ps1 -PipelineMerge
& FPGA/llm/sim/run_wide_sparse_dot.ps1 -PipelineMerge
& C:/Xilinx/Vivado/2024.2/bin/vivado.bat -mode batch -source FPGA/llm/vivado/implement_registered_sparse_dot.tcl -log FPGA/llm/build/pipelined_registered64.log -journal FPGA/llm/build/pipelined_registered64.jou -tclargs 64
& C:/Xilinx/Vivado/2024.2/bin/vivado.bat -mode batch -source FPGA/llm/vivado/implement_registered_sparse_dot.tcl -log FPGA/llm/build/banked_registered64.log -journal FPGA/llm/build/banked_registered64.jou -tclargs 64 banked
.\.venv\Scripts\python.exe FPGA/llm/research/report_pipelined_finder.py
```

See [measured effects and timing](PIPELINED_FINDER_RESULTS.md). Cycles from W4 source-stall cases remain correctness-only when comparing widths, since injected bubbles are per frame. The main fixtures preserve work, source packet sizes and acquisition charges.
