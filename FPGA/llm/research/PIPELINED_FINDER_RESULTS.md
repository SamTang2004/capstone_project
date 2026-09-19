# Pipelined merger: measured effects

**The 64-position block meets the 100 MHz routed timing target between explicit shared-clock launch/capture registers.** The computation still uses eight DSP48E1 multipliers. This is static timing evidence for the routed out-of-context block, not a board measurement.

RTL correctness: **104 result vectors / 762 dot results** pass in 187,243 scoreboard cycles. The standalone finder passes **4,842 frames** / 33,320 cycles, including 16,506 output-stall cycles. Directed checks fill the pipeline, drain it, reset multiple in-flight frames, reset mid-frame, and sustain 100 consecutive sparse frames without admission bubbles. Payload, source index, metadata, empty-frame handling and full nonfinal chunks are checked.

## Routed timing and area at scan 64

| Design | LUTs | Registers | DSPs | BRAM | LUTRAMs included in LUTs | Setup WNS ns | Hold WHS ns |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| banked | 6919 | 2703 | 8 | 0 | 0 | 0.003 | 0.011 |
| pipelined | 14121 | 6068 | 8 | 0 | 728 | 0.009 | 0.031 |

The table reports DUT resources. The timing harness adds 1,938 preserved launch/capture registers around each design (1,382 input and 556 output bits). These fixture registers are excluded from DUT cycle and resource counts but included during physical implementation.

Vivado 2024.2, xc7z020clg400-1; 10.000 ns clock, 0.300 ns uncertainty, clock origin BUFGCTRL_X0Y0. All DUT inputs and outputs connect to explicit registers on the same clock. Two false-path constraints exclude only the outer fixture-port wiring, before launch registers and after capture registers. No DUT or register-to-register path is excepted and no multicycle exception is used. Timing checks report no missing clocks, unconstrained internal endpoints or loops; all routable nets are fully routed. The 1,382 missing input-delay and 556 missing output-delay warnings concern only the outer fixture ports. There are no DRC errors; DSP pipelining advisories and the standalone Zynq PS7 warning remain. Only width 64 is physically qualified here; the other widths are RTL-tested.

The previous combinational width-64 merger had 13,313 LUTs, 2,703 registers and −6.724 ns setup slack in synthesis with no IO budgets. That older estimate is not a routed comparison. The banked and pipelined rows above use the same physical flow and constraints.

Before the two-frame input FIFO, the pipelined design failed a different external-port timing experiment: setup −1.002 ns and hold −1.220 ns with 2 ns maximum / 0.5 ns minimum IO delays. The FIFO breaks the buffer-to-input-ready path. Explicit shared-clock boundary flops replace the ideal external-clock IO model in the final experiment. **The earlier external-port contract is not claimed to pass.** The final result qualifies the registered on-chip block environment; it does not qualify the timing harness fixture ports or board IO.

## Resident wide words

Complete command-to-result cycles at scan 64.

| Fixture | Banked | Combinational merge | Pipelined merge | Pipelined time at 100 MHz | Useful utilization |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense_w8 | 261 | 261 | 266 | 2.66 µs | 96.241% |
| half_w8 | 260 | 145 | 150 | 1.50 µs | 85.333% |
| ten_percent_w8 | 571 | 155 | 160 | 1.60 µs | 62.500% |
| point_one_x_99 | 133 | 133 | 138 | 1.38 µs | 0.725% |
| zero_jobs | 13 | 13 | 18 | 0.18 µs | 0.000% |
| short_jobs | 15 | 15 | 20 | 0.20 µs | 25.000% |

## Supply capped at eight operand pairs per cycle

Complete command-to-result cycles at scan 64.

| Fixture | Banked | Combinational merge | Pipelined merge | Pipelined time at 100 MHz | Useful utilization |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense_w8 | 268 | 268 | 273 | 2.73 µs | 93.773% |
| half_w8 | 268 | 265 | 270 | 2.70 µs | 47.407% |
| ten_percent_w8 | 1008 | 1005 | 1010 | 10.10 µs | 9.901% |
| point_one_x_99 | 1005 | 1005 | 1010 | 10.10 µs | 0.099% |
| zero_jobs | 69 | 69 | 74 | 0.74 µs | 0.000% |
| short_jobs | 16 | 16 | 21 | 0.21 µs | 23.810% |

## Cost versus benefit

The pipelined DUT uses **2.04x the LUTs** of the banked control. The following ratios use the complete resident-input command interval at the same 100 MHz clock; they are not memory-system or energy measurements.

| Density fixture | Speedup versus banked | Throughput per LUT versus banked |
| --- | ---: | ---: |
| half_w8 | 1.73x | 0.85x |
| ten_percent_w8 | 3.57x | 1.75x |
| point_one_x_99 | 0.96x | 0.47x |

LUT-normalized throughput omits register, DSP, memory-bandwidth and energy costs. It is a limited screening metric, not a complete hardware-efficiency claim.

## Pipeline effects

- At scan 64, the input FIFO, three registered merge levels and registered serializer add **five cycles / 50 ns** to the unstalled fixtures relative to combinational merging. Scan 8/16/32 add two/three/four cycles. This is a per-wave fill/drain increase in these fixtures, not a five-cycle penalty on every frame.
- Frames overlap across elastic stages. A frame with eight survivors can be accepted every cycle; a dense 64-entry frame still needs eight serializer cycles. Backpressure fills the finite pipeline and then stalls input without losing ownership or data.
- More registers store frames in flight. The staged merge tree also replaces the large global rank-selection network, so the area change reflects both added pipeline storage and a different compaction topology.
- Operand frames use distributed RAM in the input queue. Metadata is kept in flip-flops to avoid a short LUTRAM-output hold path. This storage choice does not change cycle latency.
- The input supply ceiling remains. The resident-mode gains require wide local operand availability; the eight-pair cap removes most gains. The extreme sparse fixture remains below 1% useful utilization.
- No power or energy result is claimed. Additional registers and clock load must be measured against saved arithmetic activity. No SRAM banking, DMA/DDR path or board validation is included.

W4 source-stall cases are correctness-only across widths because source bubbles are per frame. The report checks equal work, source packet sizes and acquisition charges, and an unchanged dense control.

[Architecture and reproduction](PIPELINED_FINDER.md) · [Raw metrics](PIPELINED_FINDER_PERFORMANCE.csv)
