# Complete sparse overlay at 200 MHz

Physical PYNQ-Z2 validation: 96 commands at 200 MHz and
32 commands at 100 MHz using the same new bitstream.
Every INT32 result, metadata map and activation request count passes.
Repeated counters are identical, and the same image has identical cycle
counts at both clocks. The host AXI bus remains at 100 MHz.

## Timing and area

Complete PS/PL implementation on xc7z020clg400-1, Vivado 2024.2:
+0.004 ns setup / +0.024 ns hold slack at 200 MHz, with 0.300 ns user
setup uncertainty on both clocks. No unconstrained internal endpoints or
routing errors. CDC reports ten synchronized crossings and no warnings;
all five bus-skew constraints pass (minimum slack 13.682 ns).

| Resource | Original 100 MHz overlay | New 200 MHz overlay |
| --- | ---: | ---: |
| LUTs, including LUTRAM/SRL | 4,204 | 4,323 |
| Fabric flip-flops | 6,564 | 7,774 |
| DSP48E1 | 8 | 8 |
| BRAM tiles | 67.5 | 67.5 |

This costs 2.8% more LUTs and 18.4% more flip-flops. The final DRC report
retains two LUT-pair warnings and eight unused-DSP-port power advisories;
bitstream generation passes. The preset DDR-DQS warnings remain historical
board-preset caveats, not a DDR throughput qualification.

## Pipeline changes

- Registered weight requests and memory output before byte selection.
- Fixed-latency weight reads with a 16-entry response queue and at most 12
  outstanding requests. Credits remove combinational backpressure from BRAM enables.
- Registered activation addresses, memory output, and aligned masks/last markers.
- Local registered upload controls at each weight bank; simplified cache read enable.
- Registered cancellation/reset distribution.
- Board-local AXI-Lite control captures AW/W before decoding, then splits
  register reads into bank selection and response stages.
- AXI3-to-Lite conversion runs at 100 MHz, followed by the generated AXI clock
  converter and the 200 MHz accelerator. Shared DeiT/GEMM RTL is unchanged.

Simulation passes 25 memory-harness commands, cancellation at all four weight
pipeline positions, and sustained core backpressure. AXI tests pass 33 writes
and 51 reads with independent AW/W arrivals, byte strobes, response stalls
and reset during a partial transaction.

## Measured effect: K=4096, 64 rows, skip + reuse

| Nominal density | Old consumer cycles | New consumer cycles | Old preparation + compute (us, 100 MHz) | New preparation + compute (us, 200 MHz) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 100% | 33565 | 33591 | 340.820 | 170.550 | 1.998x |
| 50% | 17189 | 17215 | 177.060 | 88.670 | 1.997x |
| 10% | 3963 | 3989 | 44.800 | 22.540 | 1.988x |
| 1% | 843 | 951 | 13.600 | 7.350 | 1.850x |
| 0.1% | 177 | 211 | 6.940 | 3.650 | 1.901x |
| 0% | 92 | 116 | 6.090 | 3.175 | 1.918x |

Preparation cycles increase from 517 to 519 for K=4096. Added latency
includes three extra weight-path drain cycles per eight-row tile and two
extra activation-read stages. Frame latency can overlap useful weight work
at higher densities. Pipelining preserves a sustained
one-column-per-cycle weight path but is not free for short sparse workloads.

## Scope and provenance

Operands reside in uploaded BRAM. The table includes metadata preparation
and consumer execution; it excludes CPU transfers. It is not a DDR, complete
LLM inference, or fixed-external-bandwidth speedup.

Initial weight upload at the 200 MHz run: 4.254 s.
Median host workflow for K=4096 with weights already resident: 68.499 ms.
Python/AXI-Lite upload cost dominates the current host workflow.

BIT SHA256: `6d437a06fdc245271dd81329f87d0fee70e4e2b2beb5fb41876dcc27f0c04aa0`.
HWH SHA256: `bebc8caa57f636886b84889b499b9df63f97f17ba1dc1492dcd7b9cc9ade1107`.
Release host timestamp: `2026-09-18T21:38:44.374127+00:00`.

Raw results and matched timing/resource/CDC reports are in `board200/`.
The original 100 MHz measurements remain in `board100/`.
