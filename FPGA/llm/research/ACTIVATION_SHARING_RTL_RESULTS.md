# Activation sharing: Vivado results

RTL simulation passes **129 result vectors / 768 dot products** across three architectures, using DSP48E1 models. The standalone bitmap finder passes **4225 frames / 101088 survivors**, including 51869 stalled-output cycles. [Architecture and scope](ACTIVATION_SHARING_RTL.md).

## Shared selection: command-to-result cycles

K=4096, eight output rows, signed W8 nonzero weights, deterministic random fixed-count activation masks. Eight activation bytes/cycle; eight output-interleaved weight banks, two-cycle responses. Acquisition, selection and arithmetic drain are included. No source/result stalls in this table.

| Nominal activation density | Survivors | Tree cached | Bitmap cached | Bitmap streaming | Streaming useful utilization |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100% | 4096 | 8212 | 8207 | 4110 | 99.66% |
| 50% | 2048 | 4116 | 4111 | 2062 | 99.32% |
| 10% | 409 | 938 | 951 | 541 | 75.60% |
| 1% | 40 | 565 | 560 | 519 | 7.71% |
| 0.1% | 4 | 529 | 524 | 519 | 0.77% |
| 0% | 0 | 526 | 521 | 519 | 0.00% |

## Scanning efficiency of bitmap streaming

The bound is max(ceil(K/activation_bandwidth), survivor_count). One selected activation per cycle can occupy all eight compute lanes. Scan cycles include input acquisition, finite-buffer backpressure and frame handling. Efficiency here is relative to that service bound, not the fraction of inputs retained.

| Nominal density | Scan cycles | Optimistic bound | Bound / scan cycles | Input positions scanned/cycle |
| --- | ---: | ---: | ---: | ---: |
| 100% | 4104 | 4096 | 99.81% | 0.998 |
| 50% | 2056 | 2048 | 99.61% | 1.992 |
| 10% | 535 | 512 | 95.70% | 7.656 |
| 1% | 513 | 512 | 99.81% | 7.984 |
| 0.1% | 513 | 512 | 99.81% | 7.984 |

## Sharing ablation at 10% nominal density

Eight passes repeat classification but compute the output tile only once. The same weight requests and arithmetic are used in both columns. This is a repeated-scanning ablation, not a dense baseline.

| Variant | Eight scans | One shared scan | Speedup |
| --- | ---: | ---: | ---: |
| Tree cached | 4543 | 938 | 4.84x |
| Bitmap cached | 4661 | 951 | 4.90x |
| Bitmap streaming | 4251 | 541 | 7.86x |

## Synthesis comparison

Same part, 10 ns clock and 0.300 ns uncertainty. These are **unplaced** LUT/register counts and timing estimates; external IO paths lack delay budgets. Only the separately routed streaming qualification below can establish the stated clock target.

| Variant | LUTs | FFs | BRAM36-equivalent tiles | DSPs | Setup estimate ns | Hold estimate ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tree cached | 8693 | 4307 | 2.5 | 8 | +1.246 | -0.111 |
| Bitmap cached | 1742 | 1347 | 2.5 | 8 | +0.114 | -0.037 |
| Bitmap streaming | 1738 | 1325 | 0 | 8 | +0.590 | -0.029 |

## Routed 100 MHz qualification

**Bitmap streaming passes** at 10.000 ns with 0.300 ns clock uncertainty: setup slack **+0.085 ns**, hold slack **+0.012 ns**. Routed DUT: **1735 LUTs / 1325 FFs / 8 DSPs**, 0 RAMB36 and 0 RAMB18.

All DUT ports connect to same-clock launch/capture registers. The 983 preserved fixture registers are excluded from DUT area and functional cycle counts, but included in placement. Only outer fixture-port paths are excepted. No DUT paths or register-to-register paths are false or multicycle. The expected unconstrained outer-port delay warnings do not qualify board IO. No internal missing clocks/unconstrained endpoints/loops, routing errors or DRC errors are present.

This qualifies the streaming on-chip block boundary. It does not include physical weight RAM, its response queue, DMA/DDR, production placement or board validation. The two cached variants have synthesis estimates only.

## Implications and limits

- Bitmap cached uses 80.0% fewer synthesized LUTs than tree cached for a similar cycle count. Both include the same descriptor capacity and eight DSPs.
- Streaming overlaps scanning and computation; it removes the descriptor cache for one eight-row tile. Retaining/replaying selections for later output tiles is future work.
- At 64 activation bytes/cycle, bitmap streaming takes 416 cycles at 10% density (98.32% useful utilization). This gain requires that wide input supply.
- Near 0.1% density the eight-byte source supplies only about 0.008 useful activations/cycle. The low MAC utilization reflects input sparsity and bandwidth even when scanning is nearly optimal.
- The Python model used different buffering and a different preparation throughput. These RTL numbers supersede its estimates only for this concrete eight-row, output-interleaved configuration.
- No power measurement or energy claim. No general sparse-weight compaction, multi-tile RTL reuse, real LLM quality evaluation or external-memory performance claim.
