# Pipelined activation engine at 200 MHz

The consumer and producer separately meet 5.000 ns with 0.300 ns uncertainty on xc7z020clg400-1. This qualifies registered block boundaries, not a complete board or DDR system.

## Physical results

| Unit | LUTs | Fabric FFs | BRAM36 equivalents | DSPs | Setup slack ns | Hold slack ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Consumer | 2109 | 2131 | 2.5 | 8 | +0.105 | +0.007 |
| Producer | 246 | 244 | 0 | 0 | +0.873 | +0.056 |

The previous 100 MHz consumer used 2497 LUTs / 1542 fabric FFs, eight DSPs and 2.5 BRAM tiles. The previous producer used 254 LUTs / 143 FFs. Boundary fixture FFs (1072 / 148) are excluded from these counts. Both runs have no internal unconstrained timing endpoints, routing errors, or DRC errors; only the outer fixture ports are excepted.

## Matched elapsed-time comparison

K=4096, 64 output rows, eight lanes, resident metadata, block skipping and reuse. Baseline is the immediately preceding four-stage-DSP engine at 100 MHz, not a dense baseline.

| Valid density | Old cycles at 100 MHz | New cycles at 200 MHz | Speedup, same bytes/cycle | New cycles, fixed bytes/second | Speedup, fixed bytes/second |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100% | 32860 | 33566 | 1.958x | 65652 | 1.001x |
| 50% | 16476 | 17182 | 1.918x | 32884 | 1.002x |
| 10% | 3464 | 3850 | 1.799x | 6860 | 1.010x |
| 1% | 598 | 605 | 1.977x | 1128 | 1.060x |
| 0.1% | 145 | 151 | 1.921x | 222 | 1.306x |
| 0% | 90 | 93 | 1.935x | 109 | 1.651x |

The same-bytes/cycle model doubles activation and weight bandwidth from 0.8 to 1.6 GB/s each when moving from 100 to 200 MHz. The fixed-bandwidth model doubles activation assembly/response cycles and spaces weight service by two cycles with four-cycle minimum response latency, keeping both supplies at 0.8 GB/s. Both use finite eight-entry response queues. These are simulated local interfaces, not DDR measurements.

Only resident-metadata consumer time is compared here. The new full-rate producer needs 515 cycles instead of 513; its source was not bandwidth-throttled in the fixed-bandwidth run. Cold metadata preparation must therefore be modeled separately before drawing end-to-end conclusions.

## What changed

- Block request generation separates group selection from eight-bit lane selection and registers the request.
- The finder registers magnitudes, threshold decisions, group occupancy, group payload selection and survivor output. Local lane traversal sustains one survivor/cycle; group transitions and frame preparation add bubbles.
- The tiled broadcast core uses fixed lane-to-row assignments at elaboration, eliminating the unnecessary general cross-lane reduction. The generic tagged core retains its original mapping by default.
- Empty-map detection runs after command-map capture in the existing START stage.
- The producer pipelines magnitude, threshold and bitmap updates. The four-stage DSP pipeline remains enabled.

No tiles are overlapped: the controller still waits for each result handshake before starting the next tile. The frequency gain must be weighed against added selection cycles and actual memory bandwidth.

## Validation

Each memory model passes 76 complete commands, 336 tile results and 2548 dot results, plus 32768 producer threshold cases. Tests include stalls, reset, empty inputs, partial tiles, and full descriptor caches. The finder additionally passes 4232 frames / 101102 survivors and reset cancellation at seven pipeline depths. Three shared-activation variants pass 129 vectors / 768 dots. Physical verification checks DSP registers, setup/hold, exceptions, timing coverage, routing and DRC.

Reproduce the simulations and routing using [TILED_ACTIVATION.md](TILED_ACTIVATION.md), then run `python FPGA/llm/research/report_pipeline_200.py`. Baseline input data is preserved in [PIPELINE_200_BASELINE.csv](PIPELINE_200_BASELINE.csv).

## Device limits

AMD DS187 tables 66/67 specify 464.25 MHz for a fully registered DSP48E1 and 464 MHz for the global clock tree on the -1 speed grade. These component limits are not a promise that arbitrary logic meets that frequency. Our achieved frequency is determined by routed register-to-register paths. [AMD DS187](https://docs.amd.com/api/khub/documents/uaBd8Qxmf_K1~~tiVnOxrw/content).
