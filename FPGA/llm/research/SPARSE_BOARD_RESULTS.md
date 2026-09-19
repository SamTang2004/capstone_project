# Sparse engine: physical PYNQ-Z2 results

96 commands passed exact signed INT32 golden checks at 100 MHz.
Metadata maps and activation request counts also passed. Hardware cycle counts
are identical across repeated cases.

- Board: `pynq`; ABI `0x53505202`.
- Release host timestamp: `2026-09-18T20:49:39.555485+00:00`.
- BIT SHA256: `32108df688e5ae9c125489f725ca6e9cce368f6d8ef649337e2991b5939d0bb7`.
- HWH SHA256: `706ed4cb5dc234cc43c49c7c501552b88c961a06c7c01ad4a5e7e0bb98689805`.

## K=4096, 64 output rows

All modes use the same signed operands and eight DSP lanes. Density is the
nominal generator setting; the survivor column reports actual nonzero values.
Each cell is consumer execution cycles. Preparation is a separate measured
pass required by this harness, including when skipping is disabled.

| Density | Survivors | Raw rescan | Skip only | Reuse only | Skip + reuse | Combined + preparation (us) | Raw / combined, including preparation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100% | 4096 | 38584 | 38584 | 33565 | 33565 | 340.820 | 1.147x |
| 50% | 2049 | 22208 | 22208 | 17189 | 17189 | 177.060 | 1.283x |
| 10% | 412 | 8072 | 8072 | 3963 | 3963 | 44.800 | 1.917x |
| 1% | 42 | 5768 | 3832 | 1085 | 843 | 13.600 | 4.621x |
| 0.1% | 5 | 5760 | 576 | 825 | 177 | 6.940 | 9.045x |
| 0% | 0 | 5760 | 120 | 797 | 92 | 6.090 | 10.307x |

## Measurement limits

Operands are uploaded by CPU through AXI-Lite into 4 KiB activation and 256 KiB
weight BRAM storage. CPU upload and result retrieval are excluded from the
hardware cycle counts. JSON separately records initial weight upload time and
per-command activation upload, command/polling, result retrieval, and workflow
wall times (with weights already resident). These results
are not DDR bandwidth, complete LLM inference, energy, or frontier comparisons.

Initial 256 KiB weight upload: 4.224 s. For K=4096,
median activation upload: 65.943 ms;
median complete command workflow with resident weights: 69.642 ms.
Python/AXI-Lite transfers dominate this host-driven workflow, so the hardware
cycle gains do not imply comparable application wall-time speedups.

Raw rescan is the same sparse consumer with skip/reuse disabled, not an
independently optimized dense GEMM. Preparation would be avoidable in a
production no-skip path; charge that distinction in future architecture comparisons.

At 0.1% density, combined preparation + consumption is 694 cycles (6.94 us).
Reuse-only consumption takes 825 cycles: if its unnecessary metadata pass is
removed, the combined approach is only 1.19x faster than that stronger baseline.
At 1% density, combined 1,360 cycles is slower than reuse-only consumption
(1,085 cycles). An adaptive choice or metadata generated upstream remains useful.

## Complete implementation

Vivado 2024.2, xc7z020clg400-1, 100 MHz, 0.300 ns user setup uncertainty:
+1.017 ns setup / +0.036 ns hold slack, no unconstrained internal endpoints,
and no routing errors. Total: 4,204 LUTs (including 60 SRLs), 6,564 fabric FFs,
eight DSPs, 66 RAMB36 + three RAMB18 (67.5 BRAM tiles). Bitstream generation
passes. DRC retains eight LUT-pair warnings and eight unused-DSP-D-port power
advisories; the board preset also emits inherited DDR-DQS delay warnings.

The complete 200 MHz integration failed routed setup at -1.947 ns and was not
programmed. The earlier isolated core passed 200 MHz, but this board harness
adds memory and control paths. The worst 200 MHz path crosses from survivor
cache BRAM to weight BRAM addresses; register that boundary before retrying.

Reproduce with `board/DEPLOYMENT.md`. Preserve the raw JSON, stdout, routed
reports and release manifest alongside this summary.
