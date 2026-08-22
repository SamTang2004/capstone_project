# Ordered Loader Cycle-Limit Analysis

This analysis uses the second independent 100 MHz board run in `run_2`. The
firmware layer time was 52.020 ms, within 0.002 ms of the first 52.019 ms run.
The bottleneck is therefore deterministic PL behavior rather than host or
Ethernet jitter.

## Main Conv2D Phase Occupancy

The table aggregates Conv layers 0, 4, and 8.

| Phase | Cycles | Share of Conv cycles |
| --- | ---: | ---: |
| Load A | 1,152,989 | 43.90% |
| Accumulate | 536,640 | 20.43% |
| Load B | 386,559 | 14.72% |
| Write C | 276,384 | 10.52% |
| Bias preload | 148,257 | 5.64% |
| Systolic compute | 112,320 | 4.28% |
| Controller overhead | 9,190 | 0.35% |
| Total | 2,626,499 | 100.00% |

At 256 INT8 MACs per cycle, the model's 13,271,040 main-Conv MACs require
51,840 ideal cycles. This is 1.97% of measured Conv time. Even inside compute
windows, useful MAC occupancy is about 46.15% because edge tiles, K tails,
systolic fill, drain, and pipeline latency remain.

## Loader and DDR Evidence

| Counter | Value |
| --- | ---: |
| Loader address cycles | 1,093,889 |
| Loader request cycles | 190,393 |
| Loader receive cycles | 1,271,193 |
| Cache hits | 178,025 |
| Cache misses | 15,264 |
| Cache hit rate | 92.10% |
| HP0 line-fill beats | 122,112 |
| Requested bytes delivered | 1,026,700 |

Every miss used exactly eight accepted HP0 beats: `15,264 * 8 = 122,112`.
The raw line-fill channel is therefore behaving correctly. However, the loader
delivered only 0.808 requested bytes per receive cycle, or 80.8 MB/s at 100 MHz.
That is 20.2% of the 32-bit 400 MB/s target and 10.1% of the physical 64-bit
HP0 payload ceiling.

The 193,289 logical cache requests average only 5.31 useful bytes each. The
memory driver gates `read_req_ready` with `!read_data_valid_r`, so it cannot
continuously admit the next short descriptor while a prior response is valid.
Cache lookup is also held while a fill or write is active. Per-descriptor
admission, lookup, and response bubbles therefore dominate despite the high hit
rate and full-rate AXI burst beats.

## Key Limitation

The present limit is a serialized internal transaction pipeline, not raw DDR
beat bandwidth and not address arithmetic alone:

1. The controller runs Load A, Load B, compute, accumulate, and writeback as
   mutually exclusive phases.
2. The read cache cannot accept requests continuously while returning data.
3. Short im2col descriptors repeatedly pay request and lookup transition costs.
4. The array consequently computes for only 4.28% of controller time.

The next optimization should decouple request admission from response valid,
allow hit lookup and queued request intake during response delivery, and add a
response metadata FIFO that preserves order. After that, overlap next-tile A/B
loads with current-tile compute and accumulation using separate wave-buffer
banks. Increasing AXI burst width again will have little effect until this
internal serialization is removed.
