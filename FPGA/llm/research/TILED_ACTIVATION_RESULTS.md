# Block skipping and multi-tile reuse: Vivado results

**Both optimizations are implemented and verified.** 76 complete matrix-vector commands, 336 tile results and 2548 dot results pass DSP48E1 simulation. The totals include one correct tile before an intentional replay reset. The producer also passes 32,768 exhaustive byte/lane/threshold cases. [Architecture and contracts](TILED_ACTIVATION.md).

## Latency for 64 output rows

At 200 MHz: K=4096, eight compute lanes, eight-byte/cycle activation supply, nonzero signed W8 weights and deterministic random fixed-count activation masks. Every mode computes all 64 rows. Resident-map latencies include a one-cycle/eight-byte metadata read. Cold-map latencies also charge the full 515-cycle metadata-building pass. These interfaces scale bytes/second with clock; see [fixed-bandwidth comparison](PIPELINE_200_RESULTS.md) for the separate constant-bandwidth experiment.

| Nominal density | Raw rescan per tile | Reuse only | Skip + reuse, resident map | Resident speedup | Skip + reuse, separate metadata pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100% | 38584 | 33565 | 33566 | 1.15x | 34081 |
| 50% | 22200 | 17181 | 17182 | 1.29x | 17697 |
| 10% | 7328 | 3849 | 3850 | 1.90x | 4365 |
| 1% | 4248 | 881 | 605 | 7.02x | 1120 |
| 0.1% | 4248 | 629 | 151 | 28.13x | 666 |
| 0% | 4248 | 608 | 93 | 45.68x | 608 |

**Metadata preparation is charged separately.** For 0.1% density, resident-map skipping plus reuse takes 151 cycles, but building the map first takes 666 total cycles. Reuse without a map takes 629. Generate summaries alongside existing activation production, or amortize their construction over further consumers.

## Four-way ablation at 0.1% / 64 rows

| Skip | Reuse | Cycles, resident map | Activation bytes | Metadata read bytes | Weight bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| False | False | 4248 | 32768 | 0 | 256 |
| True | False | 417 | 2048 | 8 | 256 |
| False | True | 629 | 4096 | 0 | 256 |
| True | True | 151 | 256 | 8 | 256 |

These are activation-source reads, not all memory traffic. The combined mode also writes four 20-bit descriptors and reads them 28 times for later tiles (80 write bits / 560 read bits). A cold metadata pass adds 4096 activation read bytes and an eight-byte metadata write. No energy or DDR-transaction reduction is inferred from these counts.

## Single-tile and distribution limits

| Case | Raw baseline cycles | Combined resident cycles | Combined cold-map cycles |
| --- | ---: | ---: | ---: |
| Eight rows, 10% density | 916 | 917 | 1432 |
| Eight rows, 1% density | 531 | 255 | 770 |
| Eight rows, 0.1% density | 531 | 53 | 568 |
| 64 rows, clustered 1% | 4248 | 422 | 937 |

The random 1% fixture occupies 29 blocks; its clustered counterpart occupies one. The same survivor count therefore gives different skip benefits. Dense/high-density inputs occupy every block, so summaries save no reads and incur their metadata access.

## Physical implementation

Both units separately pass **200 MHz** on xc7z020clg400-1 with 0.300 ns clock uncertainty between explicit shared-clock boundary registers.

| Unit | Routed LUTs | FFs | BRAM36-equivalent tiles | DSPs | Setup slack ns | Hold slack ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Multi-tile consumer | 2109 | 2131 | 2.5 | 8 | +0.105 | +0.007 |
| Metadata producer | 246 | 244 | 0 | 0 | +0.873 | +0.056 |

The consumer grows from the previous one-tile streaming block's 1,735 LUTs / 1,325 FFs / zero BRAM to 2109 LUTs / 2131 FFs / 2.5 BRAM tiles. The producer costs an additional 246 LUTs / 244 FFs. These are separately placed blocks, not an integrated system resource/timing result. Activation/weight memories, their queues, metadata storage between units, DMA and DDR remain outside these counts.

The consumer and producer timing fixtures add 1072 and 148 preserved boundary FFs, respectively, excluded from DUT area and functional cycle counts. Only outer fixture-port paths are excepted. Both have no internal missing clocks, unconstrained endpoints, loops, routing errors or DRC errors. DSP pipelining/standalone PS7 warnings and expected outer-port delay warnings are not board IO qualification.

## Interpretation

- These results use the four-stage DSP product pipeline, pipelined bitmap selection and producer, and fixed lane-to-row arithmetic. See [200 MHz comparison](PIPELINE_200_RESULTS.md) for the 100 MHz baseline.
- Reuse removes repeated scans after tile 0. Block summaries remove empty-block reads from that first scan when metadata is already available. They address different costs.
- At high density, scanning overlaps useful computation; caching saves activation-source reads but gives little latency gain. It still consumes BRAM and descriptor traffic.
- At extreme sparsity, startup/drain for each eight-row tile becomes a substantial residual cost. The prototype deliberately waits for each result handshake before starting the next tile.
- No quality claim for positive pruning thresholds, no power measurement, no board execution, and no full LLM inference or end-to-end memory-system speedup claim.
