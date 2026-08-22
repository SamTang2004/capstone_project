# JIT Conv Address Generator Board Comparison

## Conditions

- Board: PYNQ-Z2, Zynq-7020
- Model: `demo/models/fruit3_budgetcuda_int8.onnx`
- PL clock: true 90.91 MHz (`FCLK0_CTRL=0x00100b00`)
- Array: one 16x16 INT8 systolic array
- Weight path: host-presliced, cache depth 8
- Diagnostic run: one warmup followed by one measured inference
- Baseline: `conv_profile_cache8_true90mhz`

## Results

| Measurement | Baseline | JIT address generator | Change |
| --- | ---: | ---: | ---: |
| Firmware layer time | 174.126 ms | 137.339 ms | -21.1% |
| Host execute round trip | 188.615 ms | 153.216 ms | -18.8% |
| Aggregate Conv cycles | 10,640,148 | 7,296,323 | -31.4% |
| Loader address cycles | 3,901,943 | 1,655,560 | -57.6% |
| Loader receive cycles | 1,682,231 | 1,681,553 | -0.04% |
| Conv output logits | `[-25, 33, -20]` | `[-25, 33, -20]` | exact |

The nearly unchanged receive count shows that the reduction came from address
and control scheduling, not from moving less activation data. The descriptor
FIFO and incremental recurrence therefore solved the targeted bottleneck, but
they did not change the DDR service time or the other CNN operators.

## Stress Result

A separate balanced 300-image board run produced 99.0% accuracy and 99.33%
agreement with ONNX Runtime. Mean inference latency was 185.772 ms, p95 was
190.176 ms, and maximum latency was 193.447 ms. The FPGA manager remained
`operating`, the BudgetCUDA service remained `active`, and FCLK0 remained at
the requested 90.91 MHz after the run.

## Timing Qualification

The integrated image routed at WNS `-0.448 ns` against the 10 ns constraint.
Board validation therefore used 90.91 MHz. The image must not be described as
timing-closed at 100 MHz until a new placement or RTL timing pass reports
nonnegative WNS and hold slack.
