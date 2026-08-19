# Conv2D Cycle and Regression Analysis

## Configuration

- Board: PYNQ-Z2, Zynq-7020
- Deployed compute: one 16x16 INT8 systolic array
- PL clock: 100 MHz (`FCLK0_CTRL=0x00200500`)
- Read cache: eight 64-byte lines (512 bytes total)
- Model output: `[-25, 33, -20]`
- Routed timing: WNS `+0.127 ns`, TNS `0`, WHS `+0.030 ns`

## End-to-End Result

| Measurement | Old bursty 100 MHz | Native AXI3, mislabeled 90 MHz | Cache-8, true 90.91 MHz | Cache-8, 100 MHz |
| --- | ---: | ---: | ---: | ---: |
| Firmware layers | 221.046 ms | 432.104 ms | 174.126 ms | 158.508 ms |
| Execute round trip | 223.581 ms | 446.872 ms | 188.615 ms | 172.733 ms |
| Host total | 248.026 ms | 485.120 ms | 201.110 ms | 174.521 ms |
| Conv2D total | 163.519 ms | 313.869 ms | 117.081 ms | 106.669 ms |

The apparent regression was dominated by a clock programming bug. The old
`FCLK0_90_MHZ` value was `0x00200b00`: DIVISOR1=2 and DIVISOR0=11, so the PL
ran at 45.45 MHz, not 90.91 MHz. The corrected value is `0x00100b00`.

The eight-line cache also removes line thrashing. At the same incorrect
45.45 MHz setting it reduced firmware time from 432.104 ms to 337.511 ms and
Conv2D time from 313.869 ms to 231.634 ms. The corrected 100 MHz result now
beats the earlier 223.581 ms execute result by 22.7%.

## Exact Conv2D Cycles

Across all four Conv2D layers, hardware counted 10,662,956 cycles:

| Controller section | Cycles | Conv share |
| --- | ---: | ---: |
| Load A | 4,924,576 | 46.184% |
| Write C | 2,386,564 | 22.382% |
| Accumulate | 1,066,496 | 10.002% |
| Load B | 910,111 | 8.535% |
| Zero A | 533,248 | 5.001% |
| Zero B | 533,248 | 5.001% |
| Bias preload | 189,108 | 1.774% |
| Systolic compute wait | 110,399 | 1.035% |
| Controller overhead | 9,206 | 0.086% |

The loader's overlapping internal breakdown is:

| Loader section | Cycles | Conv share |
| --- | ---: | ---: |
| Address/coordinate generation | 3,901,943 | 36.593% |
| Receive/deliver | 1,699,038 | 15.934% |
| Request acceptance wait | 210,793 | 1.977% |

Cache statistics were 216,398 hits and 15,911 misses, a 93.15% request hit
rate. Misses fetched 1,018,304 bytes in 254,576 AXI beats, while consumers
received 909,144 bytes. DDR read amplification is therefore about 1.12x; cache
capacity is no longer the primary read bottleneck.

## Conclusion

The systolic array itself is not limiting this CNN. Its compute-wait state is
only 1.035% of Conv cycles. The next optimization target is activation address
generation and A-tile construction, followed by writeback and the serial
256-entry accumulator transfer. Adding substantially more cache lines should
have diminishing returns after the measured 93.15% hit rate. A better next
step is to generate contiguous im2col runs or cache decoded activation windows,
then overlap A loading, accumulation, and writeback with adjacent tiles.
