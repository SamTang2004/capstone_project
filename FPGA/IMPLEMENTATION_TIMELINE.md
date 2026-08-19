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
