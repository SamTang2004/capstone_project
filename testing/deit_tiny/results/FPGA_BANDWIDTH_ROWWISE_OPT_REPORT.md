# W4 bandwidth, write-overlap, and rowwise optimization report

This report records the timing-qualified physical PYNQ-Z2 run after adding the
wide W4 BRAM cache path, overlapping fused GEMM writeback with subsequent
tile work, and streamlining the shared LayerNorm/Softmax engine. The complete
DeiT-Tiny model, its weights, descriptor, and workspaces remain resident in the
64 MiB DDR arena; inference uses one input transfer, one graph dispatch, and
one output transfer.

## Correctness and timing signoff

- Layer-0 patch-embedding output is byte-exact across all 37,632 bytes. Its
  SHA-256 is `51e04cbb62a4699a0b60178fa5422dd0e4c1cb57060fc5c72a49772cf33e6871`,
  identical to the known-good overlay.
- Full-model logits are exactly `[2, -2, 2, 0, 0, -5]`; the predicted class is
  `apple_fresh`.
- All 25 RTL simulations pass, including multi-N/multi-M W4 cache reuse and a
  read/final-write collision regression.
- The image passes 100 MHz signoff with WNS +0.027 ns and WHS +0.032 ns, zero
  failed routes, and no DRC errors.
- Implementation uses 13,570 LUTs (25.51%), 11,759 registers (11.05%), 39.5
  BRAM tiles (28.21%), and 153 DSPs (69.55%).

## Physical result

The exact device window is 109,571,028 cycles, or 1.09571028 seconds at
100 MHz. This is 5.69% faster than the prior exact full-fusion baseline of
116,181,263 cycles. Host-observed inference is 1.2769 seconds, of which the
single full-device graph dispatch is 1.1844 seconds. The result remains 2.19x
above the 500 ms device-window target.

| Module | Layers | Baseline cycles | Optimized cycles | Optimized time | Change |
|---|---:|---:|---:|---:|---:|
| GEMM | 218 | 50,624,076 | 45,083,301 | 450.83 ms | -10.94% |
| Softmax | 36 | 30,135,188 | 28,748,627 | 287.49 ms | -4.60% |
| LayerNorm | 25 | 19,202,424 | 19,215,665 | 192.16 ms | +0.07% |
| GELU | 12 | 6,896,474 | 7,079,040 | 70.79 ms | +2.65% |
| Residual add | 24 | 4,787,155 | 4,909,195 | 49.09 ms | +2.55% |
| Token concatenation | 12 | 3,465,607 | 3,464,303 | 34.64 ms | -0.04% |
| Token embedding | 1 | 1,070,339 | 1,070,897 | 10.71 ms | +0.05% |

Module counters are exact windows, while stall counters overlap them and must
not be summed as independent execution time. GEMM remains the largest module
at 41.15% of the full device window, followed by Softmax at 26.24% and
LayerNorm at 17.54%.

## GEMM phases

| GEMM phase | Baseline cycles | Optimized cycles | Optimized time | Change |
|---|---:|---:|---:|---:|
| Load A | 6,340,856 | 6,584,754 | 65.85 ms | +3.85% |
| Load B | 17,622,411 | 16,675,309 | 166.75 ms | -5.37% |
| Systolic compute | 10,599,617 | 10,599,617 | 106.00 ms | 0.00% |
| Fused requantize/writeback | 15,036,955 | 13,698,641 | 136.99 ms | -8.90% |

These phase counters overlap after operator pipelining. The 64-bit W4 cache
reader reduces B traffic, while writeback overlap removes serialization from
the GEMM window. Across the complete run, memory stalls still occupy
52,212,656 overlapping cycles (47.65% of the device window).

## Correctness discrepancy and fix

The first optimized image initially produced incorrect logits
`[2, -3, 1, 2, 1, -6]`. Raw prefix capture localized the first mismatch to
patch-embedding row 9, column 4: columns 4-7 of later rows were zero.

When a read request arrived while the fused writer accepted the final word of
a two-word row, the normal path scheduled a two-word burst. In the same clock,
the read-forced partial-flush path overwrote the burst length with one. The
second word was stored in the coalescing buffer but never sent to DDR. The
driver now forces a partial flush only when no writer word is valid, preserving
writer priority and the complete burst. The dedicated regression holds a read
request while the contiguous final writer word handshakes and verifies that
both words reach memory.

The fix also improves performance relative to the faulty measurement: GEMM
falls by 6.11%, B load by 14.61%, and writeback by 20.41%, because rows are no
longer split into unnecessary single-word transactions.

## Next bottleneck

The 500 ms target requires another 59,571,028 cycles of reduction. GEMM,
Softmax, and LayerNorm together account for 93,047,593 cycles (84.92% of the
window). The next work should therefore focus on DDR efficiency shared by
these modules: wider or longer activation bursts, persistent activation tiles,
and overlap between rowwise reads, arithmetic, and writes. A higher Fmax can
help after these paths retain timing margin, but at the current cycle count the
design would require about 219 MHz to reach 500 ms without further cycle
reduction.

Raw cycle data is in
[`fpga_w4a8_bandwidth_rowwise_fixed_report.json`](fpga_w4a8_bandwidth_rowwise_fixed_report.json).
The known-good and corrected layer captures are
[`bandwidth_baseline_layer0_raw.json`](bandwidth_baseline_layer0_raw.json) and
[`bandwidth_fixed_layer0_raw.json`](bandwidth_fixed_layer0_raw.json).
