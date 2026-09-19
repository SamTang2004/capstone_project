# Activation sharing: architectural model results

**Exploratory Python model, not RTL or board measurements.** All modeled outputs match independent scalar INT32 matrix-vector results. See [model contract](ACTIVATION_SHARING.md).

## Same memory layout and product order: sharing alone

K=4096, eight output rows, eight activation bytes/cycle, eight single-read weight banks, row-major W8 weights with no zeros, random fixed-count activation masks, seed 17. Phases are sequential; all activation preparation is charged.

| Activation density | Dense cached cycles | Repeated finder | Shared replay | Sharing speedup | Shared useful utilization |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100.0% | 4615 | 8303 | 4628 | 1.79x | 88.50% |
| 50.0% | 4615 | 6329 | 2682 | 2.36x | 76.36% |
| 10.0% | 4615 | 4647 | 1021 | 4.55x | 40.16% |
| 1.0% | 4615 | 4239 | 613 | 6.92x | 6.69% |
| 0.1% | 4615 | 4167 | 541 | 7.70x | 0.74% |
| 0.0% | 4615 | 4149 | 523 | 7.93x | 0.00% |

## Reuse scaling at 10% activation density

Same row-major configuration. Rows above eight execute as sequential eight-context waves; the shared descriptors persist across waves.

| Output rows | Repeated cycles | Shared cycles | Speedup | Activation source bytes, repeated / shared |
| --- | ---: | ---: | ---: | ---: |
| 1 | 587 | 587 | 1.00x | 4096 / 4096 |
| 8 | 4647 | 1021 | 4.55x | 32768 / 4096 |
| 32 | 18588 | 2530 | 7.35x | 131072 / 4096 |
| 64 | 37176 | 4542 | 8.18x | 262144 / 4096 |

## Layout and broadcast at eight output rows / 10% density

Both weight layouts are pre-existing model inputs. Transposing or reloading a matrix is NOT free and is not modeled. Compare replay/broadcast within each layout before comparing layouts.

| Layout | Shared replay cycles | Shared broadcast cycles | Descriptor read bytes, replay / broadcast | Weight bytes (both) |
| --- | ---: | ---: | ---: | ---: |
| row_major | 1021 | 1078 | 9840 / 1230 | 3280 |
| output_interleaved | 3364 | 935 | 9840 / 1230 | 3280 |

## Sensitivity at eight rows / 10% density

Same row-major layout; ranges cover seeds 17, 29 and 43. Bank-clustered masks concentrate surviving indices in one weight bank. Wider activation supply reduces the preparation cost available to amortize.

| Activation pattern | Activation bytes/cycle | Shared replay speedup over repeated finding |
| --- | ---: | ---: |
| random | 8 | 4.55–4.55x |
| random | 64 | 1.97–2.00x |
| clustered | 8 | 4.85–4.85x |
| bank_clustered | 8 | 1.95–1.95x |

## Bounds and limitations

- If repeated preparation and compute overlapped perfectly, its optimistic bound is 4144 cycles, versus 1021 for sequential shared replay (4.06x). This is a sensitivity bound, not an implemented baseline.
- The 10% / eight-row shared descriptor store is 1230 logical bytes (index plus activation); physical banking, replicated ports and BRAM rounding are uncosted.
- Repeated finding and shared replay issue exactly the same weight requests and useful MACs. Sharing alone saves activation reads and classification, not weight bytes.
- Source bytes mean local operand-memory reads, not DDR transactions. Descriptor traffic is reported separately. Weight loading, bursts, DMA, row-result writes and external bandwidth are outside this model.
- Dense control caches the activation once and uses the faster of replay/broadcast schedules for each layout. A shared dense bypass therefore remains attractive at high density.
- No LUT, power, synthesis, or 100 MHz qualification is claimed for the proposed sharing frontend. The previous finder timing result does not transfer to this architecture.
- Full CSV contains 1200 configurations, including three seeds, clustered masks, bank-clustered masks, 8/64-byte activation supply, two layouts, and 1/8/32/64 output rows.
