# Timing-Closed 100 MHz CNN Diagnostic Analysis

## Conditions

- Board: PYNQ-Z2, Zynq-7020
- Model: `demo/models/fruit3_budgetcuda_int8.onnx`
- PL clock: timing-closed 100 MHz production image
- Array: one 16x16 INT8 systolic array
- Weight path: host-presliced, eight-line DDR read cache
- Instrumented sample: one warmup and one measured inference per run
- Repetition: four instrumented runs and one 100-inference load test
- Baseline: `jit_conv_addr_90mhz`, running at a true 90.91 MHz

## Repeated Results

The four firmware samples were effectively deterministic. Their mean summed
layer time was 139.642 ms. The three newly captured reports measured total
firmware times of approximately 139.74-139.76 ms including graph-level timing
overhead.

The 100-inference test passed FPGA/software-device equality and ONNX tolerance:

| Measurement | Result |
| --- | ---: |
| Mean application-visible latency | 185.506 ms |
| Median latency | 187.812 ms |
| P95 latency | 195.613 ms |
| Maximum latency | 200.222 ms |
| Worst ONNX error | 0.66003227 |

## Operator Cost

| Operator | Aggregate time | Share of layer time |
| --- | ---: | ---: |
| Conv2D GEMM | 77.780 ms | 55.7% |
| Requantize | 30.569 ms | 21.9% |
| Pool | 20.633 ms | 14.8% |
| Activation | 10.653 ms | 7.6% |
| Reshape | 0.008 ms | less than 0.1% |

Making only the systolic arithmetic infinitely fast would save about 1.1 ms,
because the measured compute-wait state is a small part of Conv. Non-Conv
operators already consume about 61.9 ms, while host-visible input, command, and
output transactions add roughly another 30-45 ms before retaining any of the
remaining Conv memory and controller cost.

## Conv Controller Cost

The four-run mean Conv counters were:

| Phase | Cycles | Conv share |
| --- | ---: | ---: |
| Write output | 2,386,564 | 30.7% |
| Load A | 1,931,049 | 24.8% |
| Zero A and B | 1,066,496 | 13.7% |
| Accumulate | 1,066,496 | 13.7% |
| Load B | 971,919 | 12.5% |
| Bias | 231,970 | 3.0% |
| Systolic compute wait | 110,399 | 1.4% |

The array's compute state is only 1.4% of Conv controller time. Raising the
clock therefore accelerates a large amount of memory and control work rather
than turning the design into a compute-bound accelerator.

## Why 100 MHz Did Not Produce a Large Latency Gain

The timing-closure change registered the read request, associative cache
lookup, and lookup result. This removed the cache/mux cone from the critical
path, but `read_req_ready` remains low while any lookup, fill, or delivery is
active. The stages improve maximum frequency but cannot accept a new request
each cycle; their latency is paid serially for every request.

Compared with the 90.91 MHz JIT baseline:

| Counter | 90.91 MHz | 100 MHz | Change |
| --- | ---: | ---: | ---: |
| Conv cycles | 7,296,323 | 7,774,133 | +6.5% |
| Load A cycles | 1,573,000 | 1,930,927 | +22.8% |
| Load B cycles | 901,068 | 972,100 | +7.9% |
| Bias cycles | 188,470 | 231,945 | +23.1% |
| Loader receive cycles | 1,681,553 | 2,120,093 | +26.1% |
| Cache hits | 216,398 | 216,398 | unchanged |
| Cache misses | 15,911 | 15,911 | unchanged |
| Bytes delivered | 909,144 | 909,144 | unchanged |

The unchanged traffic and cache counters prove that the model did not move
more data. It paid more cycles to move the same data. The 10% frequency increase
was largely cancelled by the 6.5% Conv cycle increase and similar per-request
latency added to requantization, activation, and pooling.

The host library also opens a new TCP connection for each write, execute, and
read transaction. This is why application-visible latency remains around
185.5 ms while firmware layer execution is around 139.7 ms.

## Required Fixes

1. Make the cache path throughput-pipelined, not merely latency-pipelined. Add
   a small request FIFO and owner/tag FIFO so cache lookup can accept another
   request while a prior hit is being delivered. Misses can still serialize.
2. Issue larger read units from requantization, activation, and pooling. Their
   current fine-grained request/response loops repeatedly pay lookup latency.
3. Fuse Conv output, requantization, and activation. Conv currently spends
   23.9 ms writing INT32 accumulators, after which requantization spends another
   30.6 ms reading them and writing INT8 values.
4. Remove mandatory tile zeroing where a tile is fully overwritten, or use
   validity epochs. Zero A/B consumes another 10.7 ms.
5. Keep a persistent TCP connection or add a combined input-write/execute/
   output-read command. This targets host-visible overhead, not PL cycles.

The highest-value route below 100 ms is therefore Conv-requant-activation
fusion plus a throughput-pipelined memory front end. Increasing FCLK alone
cannot overcome the current serialized transactions.
