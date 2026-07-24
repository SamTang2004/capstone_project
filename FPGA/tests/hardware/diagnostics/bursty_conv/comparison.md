# Bursty Conv2D TileLoader Diagnostic Comparison

## Conditions

- Board: PYNQ-Z2
- PL clock: 100 MHz
- Model: fruit CNN, 17 lowered layers
- Input seed: 2026
- Warmup runs: 1
- Output logits before and after: `[-25, 33, -20]`
- Bursty bitstream SHA-256: `12A1AC88EF9D01AD5CDD63A627A41B558C3585961E4866AC61EB81A692F9BA9E`

## End-to-End Results

| Metric | Scalar loader | Bursty loader | Change |
| --- | ---: | ---: | ---: |
| Firmware layer time | 502.753 ms | 221.046 ms | -56.03% |
| Host total time | 537.833 ms | 248.026 ms | -53.88% |
| Execute round trip | 514.968 ms | 223.581 ms | -56.58% |
| Host input write | 11.861 ms | 12.677 ms | +6.88% |
| Host output read | 11.001 ms | 11.763 ms | +6.93% |
| Arena high-water usage | 565,568 B | 565,568 B | 0.00% |

## Conv2D Results

| Layer | Scalar time | Bursty time | Time reduction | Estimated utilization gain |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 91.498 ms | 48.151 ms | 47.37% | 1.90x |
| 4 | 189.951 ms | 67.752 ms | 64.33% | 2.81x |
| 8 | 163.691 ms | 47.555 ms | 70.95% | 3.44x |
| 14 | 0.135 ms | 0.061 ms | 55.18% | 2.24x |
| All Conv2D | 445.275 ms | 163.519 ms | 63.28% | - |

The unchanged non-Conv2D layer times isolate the measured speedup to the new
Conv2D loading path. Host transfer variation is small relative to the 281.707 ms
firmware reduction.

## Timing Qualification

The build completed and ran correctly, but it does not formally close timing at
100 MHz. Routed WNS was -1.016 ns and post-route physical optimization improved
it to -0.919 ns, with TNS -214.693 ns. The reported worst path ends in a
systolic PE LUT multiplier, not in the TileLoader. This is a measured
performance win, but the implementation remains provisional until timing closes.
