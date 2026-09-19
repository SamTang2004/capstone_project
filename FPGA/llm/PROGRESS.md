# LLM research progress — 2026-09-18

## Verified results

- The **complete sparse overlay now runs on the PYNQ-Z2 at 200 MHz** (host AXI
  remains at 100 MHz). Routed setup +0.004 ns / hold +0.024 ns with 0.300 ns user
  setup uncertainty; CDC and all five bus-skew constraints pass. Physical testing
  passes 96 commands / 4,920 dot results at 200 MHz and 32 commands / 1,640 dots
  with the same image at 100 MHz. Exact numerical results and all cycle counters
  match across clocks; repeated 200 MHz counters are identical. The fixes register
  weight requests/output, activation addresses/output, bank-local uploads and
  reset, replace ready propagation with response credits, simplify cache enable,
  pipeline the board-local AXI register bank, and move protocol conversion to the
  host clock. Total: 4,323 LUTs / 7,774 FFs / eight DSPs / 67.5 BRAM tiles, versus
  4,204 / 6,564 / eight / 67.5 previously. Preparation increases 517 to 519 cycles;
  K4096/R64 skip+reuse improves dense preparation+compute from 340.82 to 170.55 us
  (1.998x), 1% from 13.60 to 7.35 us (1.850x), and 0.1% from 6.94 to 3.65 us
  (1.901x). CPU transfer remains dominant (~68.5 ms median resident-weight host
  workflow), so these are on-chip execution gains, not DDR or full-LLM gains.
  Simulations pass 25 memory commands including cancellation/backpressure, 33 AXI
  writes / 51 reads, and the 76-command tiled regression plus 32,768 producer cases.
  See `research/SPARSE_BOARD_200_RESULTS.md` and raw evidence in `research/board200/`.
  The new image remains loaded at 200 MHz; old sparse 100 MHz and DeiT artifacts
  are retained and legacy services remain inactive.
- Physical PYNQ-Z2 sparse-engine deployment now passes **96 commands / 4,920 dot
  results at 100 MHz**, with exact INT32 goldens, block maps and activation-read
  counts, repeated three times with identical hardware counters. The new separate
  AXI-Lite/BRAM research overlay uses 4,204 LUTs / 6,564 FFs / eight DSPs / 67.5
  BRAM tiles; routed setup +1.017 ns / hold +0.036 ns. The complete 200 MHz attempt
  fails setup at -1.947 ns, worst from survivor-cache BRAM to weight-memory address;
  it was not programmed. Thus earlier isolated 200 MHz timing is not an integrated
  board claim. At 0.1% density, K4096/R64, skip+reuse takes 177 consumer + 517
  preparation cycles (6.94 us), versus raw rescan's 5,760 + 517 (9.04x measured
  hardware interval). Compared with reuse-only after removing its avoidable
  metadata pass, the gain is only 825/694 = 1.19x. CPU upload dominates the current
  host-driven workflow (~69.6 ms median with weights resident); no DDR/LLM speedup
  is claimed. See `research/SPARSE_BOARD_RESULTS.md`, raw evidence in
  `research/board100/`, and `board/DEPLOYMENT.md`. Old DeiT files and inactive
  services are preserved; this historical 100 MHz release is retained for rollback.
- The activation-sharing consumer and metadata producer now separately pass routed
  **200 MHz** on xc7z020clg400-1 with 0.300 ns uncertainty. Group/lane selection and
  survivor output are registered; the producer has magnitude/threshold/bitmap stages.
  The tiled core fixes each lane to its output row, removing unnecessary generic
  reduction routing. Consumer: 2,109 LUTs / 2,131 fabric FFs / eight DSPs / 2.5 BRAM,
  setup +0.105 ns / hold +0.007 ns. Producer: 246 LUTs / 244 FFs, setup +0.873 ns /
  hold +0.056 ns. Both full-rate and fixed-bandwidth simulations pass 76 commands /
  336 tiles / 2,548 dots each, plus 32,768 producer cases. Finder: 4,232 frames /
  101,102 survivors, reset at seven pipeline depths. At 0.1% / 64 rows, cycles rise
  from 145 at 100 MHz to 151 at 200 MHz: 1.92x elapsed-time gain with clock-scaled
  memory supply. Holding activation/weight bytes per second fixed gives 222 cycles,
  or 1.31x; dense inputs give only 1.001x in that fixed-bandwidth model. No board/DDR
  or integrated-system timing claim. See `research/PIPELINE_200_RESULTS.md` for the
  comparison and `research/TILED_ACTIVATION.md` for reproduction. Earlier entries
  below retain historical implementation results.
- Full DSP multiplier pipelining is enabled: AREG/BREG=2, MREG/PREG=1, with
  aligned valid/mask/slot/last metadata. All arithmetic regressions pass, plus
  65,536 exhaustive signed-byte products with bubbles and reset at every stage.
  Routed 100 MHz consumer: 2,497 LUTs / 1,542 fabric FFs / eight DSPs / 2.5 BRAM,
  setup +0.198 ns and hold +0.051 ns. DSP-output path slack improves from +1.014
  to +1.951 ns, but block-request selection limits the whole engine. A routed
  125 MHz trial fails setup at -1.186 ns (hold +0.050 ns). Eight tiles now cost
  24 additional drain cycles: the 0.1% / 64-row resident-map case is 145 cycles
  rather than 121 at 100 MHz. No whole-engine speedup is demonstrated by this
  change alone. Current tiled tables are regenerated; earlier entries below are
  historical one-stage-DSP measurements. See `research/DSP_PIPELINE_RESULTS.md`.
- Producer block summaries and **real multi-tile activation reuse** are now in RTL.
  The producer emits one threshold-specific occupancy bit per 64 activations; the
  consumer fetches occupied blocks, streams tile 0 while caching selections, then
  replays them for up to 64 output rows. Four runtime ablations compute the same
  complete matrices. Vivado DSP48E1 simulation passes 76 commands / 336 tile results /
  2,548 dots (including one verified tile before intentional reset), plus 32,768
  exhaustive producer byte/lane/threshold cases. For K=4096, 64 rows and eight-byte
  activation supply, nominal 0.1% density drops from 4,184 to 121 cycles with resident
  metadata (34.58x), and activation-source reads drop from 32,768 to 256 bytes plus
  eight metadata bytes. At 1% / 10% density, combined resident latency is 574 / 3,440
  cycles, versus 4,184 / 4,216 raw-rescan cycles. Dense inputs remain compute-bound.
  A separate 513-cycle metadata pass makes the 0.1% case 634 cycles, slower than
  reuse-only's 600: post-hoc metadata does not beat reuse alone in this sweep.
  Generate summaries during existing activation production to obtain the full gain.
  Both blocks separately pass routed 100 MHz with 0.300 ns uncertainty between
  explicit registered boundaries. Consumer: 2,497 LUTs / 1,509 FFs / eight DSPs /
  2.5 BRAM tiles, setup +0.305 ns / hold +0.010 ns. Producer: 254 LUTs / 143 FFs,
  no DSP/BRAM, setup +2.608 ns / hold +0.101 ns. Physical operand memories,
  memory queues, inter-block metadata storage, DMA/DDR and board IO remain outside
  qualification. See `research/TILED_ACTIVATION.md` and generated results/CSV.
- Activation sharing is now tested in **Vivado RTL**, including a new bitmap
  finder and streaming broadcast architecture. Three variants pass 129 vectors /
  768 dot results using DSP48E1 simulation; standalone bitmap verification passes
  4,225 frames / 101,088 survivors, including 51,869 stalled-output cycles.
  Bitmap streaming passes routed **100 MHz** between shared-clock registered
  boundaries: setup +0.085 ns / hold +0.012 ns, with 0.300 ns uncertainty.
  Routed DUT: 1,735 LUTs / 1,325 FFs / eight DSPs / no BRAM; 983 fixture FFs
  are excluded. Physical weight memory and its ordered response queue are outside
  this qualification. K=4096, eight output rows and eight activation bytes/cycle:
  50% density takes 2,062 cycles (99.32% useful utilization); 10% takes 541
  cycles (75.60%), versus 938 for tree-plus-cache. Bitmap scan efficiency is
  95.70% of the input/output service bound at 10%, 99.81% near 0.1%; the latter
  still has only 0.77% MAC utilization because raw scanning supplies little work.
  Bitmap cached uses 1,742 synthesized LUTs versus 8,693 for tree cached (80%
  reduction), both with 2.5 BRAM36-equivalent tiles. Cached write selection is
  pipelined before BRAM without adding functional cycles. Streaming overlaps
  selection and compute and eliminates the cache for one eight-row tile; reuse
  across further tiles is not implemented. See `research/ACTIVATION_SHARING_RTL.md`
  and `research/ACTIVATION_SHARING_RTL_RESULTS.md` for assumptions and reproduction.
- Activation sharing now has an executable **architectural model**, separate
  from the RTL timing results below. For K=4096 and eight output rows at 10%
  activation density, row-major shared replay takes 1,021 modeled cycles versus
  4,647 for repeated finding (4.55x), with identical weight requests. Local
  source reads fall from 36,048 to 7,376 bytes; weight reads remain 3,280 bytes.
  All 1,200 modeled configurations match scalar results; seven directed tests
  pass. The 10% / eight-row reuse gain drops to 1.97–2.00x with 64-byte
  activation supply and 1.95x for bank-clustered masks under row-major layout.
  An output-interleaved layout with activation broadcast takes 935 cycles, but
  changing layout has an unmodeled loading/conversion cost. The model charges
  activation preparation, finite request/response queues, weight bank conflicts,
  and eight-context waves. This is not new RTL or a measured 100 MHz result.
  See `research/ACTIVATION_SHARING.md` for assumptions and reproduction, and
  `research/ACTIVATION_SHARING_RESULTS.md` for the generated sweep and bounds.
- Pipelined cross-bank merger now meets routed **100 MHz** timing at scan 64
  between explicit shared-clock launch/capture registers: setup +0.009 ns,
  hold +0.031 ns, with 0.300 ns uncertainty. This qualifies the on-chip block
  boundary, not board IO or the earlier failed external-port delay contract.
  Final DUT: 14,121 LUTs (including 728 LUTRAM), 6,068 registers, eight DSPs,
  no BRAM. The same-boundary banked control also passes (+0.003/+0.011 ns),
  using 6,919 LUTs / 2,703 registers / eight DSPs. Fixture registers are separate.
  RTL passes 104 result vectors / 762 dots and 4,842 standalone finder frames.
  The two-frame input FIFO, registered merge tree and serializer add five cycles
  at scan 64. Resident 50% / 10% density takes 150 / 160 cycles (85.3% / 62.5%
  useful utilization), versus banked 260 / 571. This is 1.73x / 3.57x faster
  with 2.04x the LUT cost; the eight-pair supply cap removes the main gains.
  Metadata uses registers to eliminate a direct LUTRAM-output hold path.
  See `research/PIPELINED_FINDER.md` and `research/PIPELINED_FINDER_RESULTS.md`.
- Cross-bank merging, selectable with `MERGE_BANKS=1`: 104 result vectors /
  762 declared dot results pass in 186,956 XSim cycles; standalone 64-position
  finder passes 4,729 frames, including every tail length and reset mid-drain.
  At scan 64 with resident input, 50% density improves from 260 to 145 cycles
  (88.3% useful utilization), and 10% density from 571 to 155 (64.5%). The
  extreme 0.1%-A / 99%-B case remains 133 cycles. The eight-pair supply cap
  removes the main gains. Default bank-dispatch regression matches saved metrics.
  Merged LUTs by width: 4,038 / 4,971 / 7,896 / 13,313, all eight DSPs.
  Setup WNS: +0.651 / +0.952 / -2.028 / -6.724 ns at 10 ns; hold -0.041 ns.
  Widths 32/64 miss the setup target: evaluate pipelining before physical claims.
  See `research/MERGED_FINDER.md` and `research/MERGED_FINDER_RESULTS.md` for
  synthesis cost, full comparisons and reproducible commands.
- Wide banked finder (8/16/32/64 positions), fixed eight-DSP core: 104 result
  vectors / 762 declared dot results pass across 188,182 XSim cycles. The
  standalone 64-position finder also passes 4,662 frames / 36,968 cycles.
  Resident 0.1%-A / 99%-B fixture: 1,005 / 509 / 261 / 133 cycles by scan width.
  With supply capped at eight pairs/cycle, all four take 1,005 cycles.
  Random 10% density saturates near 571 cycles because only one nonempty bank
  drains per cycle; random 50% density sees essentially no width benefit.
  Synthesis LUTs by width: 3,967 / 4,461 / 5,372 / 7,156; all use eight DSPs.
  Width-64 setup WNS +0.256 ns / hold WHS -0.041 ns at 10 ns; no timing closure.
  See `research/WIDE_FINDER.md` and generated `research/WIDE_FINDER_RESULTS.md`.
- Eight-DSP independent dot core integrated with dense, local-finder and
  cross-packet input paths. Real DSP48E1 XSim passes 34 result vectors / 251
  declared dot results across 57,589 cycles, including W4/W8 signs, mixed-job
  reductions, zero jobs, INT32 wrap, stalls and reset cancellation.
  On eight length-1,000 dots with 0.1%-valid A and 99%-valid B, packed issue
  drops from 8,000 products to 8, but total cycles are 1,005 versus dense 1,003.
  The eight-position input scanner remains the bottleneck; no latency win.
  Final packed synthesis: 3,926 LUT, 1,763 registers, 8 DSP, no BRAM;
  setup WNS +0.749 ns / hold WHS -0.041 ns at 10 ns, not routed/timing-qualified.
  See `research/SPARSE_DOT8.md` and generated `research/SPARSE_DOT8_RESULTS.md`.
- Cross-packet work packer: finder plus bounded survivor buffer packs different
  jobs together, preserves lane/payload/job ownership, flushes partial/empty waves,
  and emits ordered job dispatch fences (not arithmetic completions). XSim passes
  9,044 scoreboard cycles; dense admission sustains eight entries/cycle in RTL.
  Completion-queue pressure drains partial packets to prevent deadlock.
  Standalone synthesis: 4,809 LUT, 1,926 registers, no BRAM/DSP. Post-synthesis
  internal setup WNS +0.804 ns, hold WHS -0.035 ns; not timing-qualified or routed.
  See `research/WORK_PACKER8.md`; no board utilization/speedup claim is made.
- Eight-lane work finder/local compactor: signed-magnitude and literal raw-bit
  threshold predicates pass XSim across 7,355 cycles / 5,345 accepted packets.
  Preserves values, payloads and original lane indices under backpressure.
  See `research/WORK_FINDER8.md`. Board integration remains future work;
  no new sparse utilization claim is made.
- GEMM-only PYNQ-Z2 build: BIT/HWH/XSA generated at 100 MHz; routed WNS
  +0.125 ns, WHS +0.036 ns. 7,486 LUT, 8,790 registers, 38 BRAM, 128 DSP.
  Old DeiT operators excluded; AXI/DDR/cache and dual GEMM cores retained.
  See `vivado/GEMM_BUILD_RESULT.md` for artifact hashes and build limitations.
- Exact sparse/dense row-fusion compiler: 143 tasks -> 50 tiles; actual DSP48E1
  XSim checks pass for 3,200 tile values and 1,049 final task outputs after
  software scatter/merge. Maximum absolute error 0. Dense-fused control is
  57 tiles, versus 330 unfused; most savings are fusion, not sparsity alone.
- Independent-dot diagonal RTL adapter: requested outputs 11 and 83 returned
  under their original lane tags. Measured execution cycles: raw separate 16,
  compacted separate 12, compacted fused 6, padded K=16 fused 20. Three input
  stalls increase fused execution to 9. Eight-lane signed and backpressure
  checks pass. Actual operand/completion trace is tracked at
  `research/diagonal_trace.jsonl`; report: `research/DIAGONAL_RTL_RESULT.md`.
- Research Python suite: 41 tests pass. Includes pre-wave S*T0 budget gating,
  exact sparsity golden checks, repeated steal-cost experiments, trace corruption
  rejection and deterministic fixture generation.

## What is NOT yet demonstrated

- No new image programmed onto the FPGA; no board latency or power measurements.
- Finder, compactor and compute prototypes now exist in RTL; there is no indexed
  sparse operand fetch engine, integrated hardware compiler or measured S*T0 deadline.
- Diagonal adapter is not in the production board manifest; it has not been
  synthesized or timing-qualified. Off-diagonal cross products remain wasteful.
- Fusion/compiler/transport cost is excluded from the 16/12/6/20-cycle comparison.
- No demonstrated same-bandwidth win against independently scheduled dot-product
  lanes or a strong sparse accelerator baseline. Wider resident-word speedups
  assume more available input bandwidth. No full TinyStories inference speedup.
- Dynamic activation sparsity is only knowable before an input-ready compute
  wave, not before all future layers/tokens. Approximate sparsity needs separate
  model-quality evaluation; the reported fusion goldens use exact zeros only.

## Next decision gate

The independent eight-lane baseline now includes input scanning, classification,
compaction and tagged reduction/accumulation in its RTL cycle comparison. Packing
reduces issued work but does not overcome the common scanner's supply limit.
The wide-finder experiment now demonstrates conditional gains with resident wide
words, lost under the eight-pair supply cap. Cross-bank merging now removes
the partial-bank drain bottleneck and improves moderate-density utilization.
The registered version now meets 100 MHz in the measured on-chip boundary,
at roughly twice the banked control's LUT cost. Next evaluate realizable
banked/indexed operand supply and integration timing, charging metadata and
transport with the same arithmetic and numerical goldens. The 10% resident
fixture improves throughput per LUT; the 50% fixture does not, so utilization
alone is insufficient to justify the cost.
Retain dense/local/packed controls and comparable memory/bandwidth budgets.
Measure useful MAC occupancy over the complete interval, area, routed Fmax and
energy; do not substitute full output beats for system-level improvement.

Working targets discussed (project goals, not industry standards): >=20%
improvement against a strong sparse baseline for a worthwhile research lead;
>=1.5x end-to-end for a competitive prototype, with 2x as an ambitious product
candidate target or a corresponding energy/cost advantage. Novelty and product
readiness require more than any single speedup number.

## Saved milestones and reproduction

- `42410fa`: timing-qualified GEMM-only release and result record.
- `9414324`: repeated work-stealing/T0 controls.
- `481c230`: pre-wave sparse fusion planner.
- `e207c95`: mixed sparse/dense compiler goldens and actual RTL trace validation.
- `146ccf7`: diagonal RTL result routing and measured cycle comparison.

```powershell
& FPGA/llm/sim/run_wide_finder.ps1 -PipelineMerge
& FPGA/llm/sim/run_wide_sparse_dot.ps1 -PipelineMerge
& FPGA/llm/sim/run_wide_finder.ps1 -MergeBanks
& FPGA/llm/sim/run_wide_sparse_dot.ps1 -MergeBanks
& FPGA/llm/sim/run_wide_finder.ps1
& FPGA/llm/sim/run_wide_sparse_dot.ps1
& FPGA/llm/sim/run_sparse_dot.ps1
& FPGA/llm/sim/run_diagonal.ps1
& FPGA/llm/sim/run_fusion.ps1
.\.venv\Scripts\python.exe -m pytest FPGA/llm/research -q -p no:cacheprovider
```

Large model/golden/build artifacts remain local and are not included in this
progress update. Generators, tests, source code, reports and the small diagonal
trace are versioned. The prior pre-DeiT-reset research checkpoint is preserved
in Git ancestry at d8c6abe. The original DeiT branch and images are unchanged.
