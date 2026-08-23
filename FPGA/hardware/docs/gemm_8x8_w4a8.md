# 8x8 Transformer GEMM tile

## Purpose

`gemm_8x8_tile_core` is the first compute block for the Transformer migration.
It calculates one output tile as a sequence of outer products:

`C[8x8] += activation[8x1] * weight[1x8]`

One accepted reduction step therefore performs 64 signed multiply-accumulates.
The accumulator is signed INT32 and wraps on two's-complement overflow.

## Numeric modes

| Mode | Activation stream | Weight stream | Decode | DSP datapath |
| --- | --- | --- | --- | --- |
| W8A8 | 8 signed INT8 lanes (64 bits) | 8 signed INT8 lanes (64 bits) | none | 64 INT8 MACs |
| W4A8 | 8 signed INT8 lanes (64 bits) | 8 packed signed INT4 lanes (32 significant bits) | sign-extend each nibble to INT8 | the same 64 INT8 MACs |

In W4A8 mode, lane 0 occupies bits `[3:0]`, lane 1 occupies `[7:4]`, and so
on. The upper 32 bits of `packed_weight_vector` are ignored. INT4 values use the
two's-complement range -8 through 7.

W4A8 is optional per tile and is intended for weight-only quantization. It
halves weight storage and weight-transfer demand, but it does not increase the
array's arithmetic throughput because values are expanded before entering the
shared DSP array. Activations and accumulators remain INT8 and INT32,
respectively.

## Transaction contract

1. Assert `start` for one cycle with `weight_int4`, `k_steps`, `valid_rows`, and
   `valid_cols` stable.
2. Wait for `input_ready`.
3. Transfer exactly `k_steps` operand pairs using `input_valid && input_ready`.
4. Wait for `result_valid`, consume the flattened 64-element INT32 result, and
   assert `result_ready`.

`valid_rows` and `valid_cols` accept values 1 through 8. Lanes outside those
limits are zeroed, which supports edge tiles without padded memory reads. Zero
`k_steps`, zero dimensions, or dimensions above eight enter the error state.

## Resource and throughput checkpoint

Out-of-context synthesis for `xc7z020clg400-1` at 100 MHz reports:

- 64 DSP48E1 blocks (29.09% of the PYNQ-Z2 device)
- 378 slice LUTs (0.71%)
- 400 slice registers (0.38%)
- 0 block RAM tiles
- 4.365 ns setup slack against a 10 ns period

At one accepted outer product per cycle, peak throughput is 64 MAC/cycle,
6.4 GMAC/s, or 12.8 GOPS when a multiply and add are counted separately. These
are compute-core figures; full-system throughput will depend on tiling, DDR
traffic, weight reuse, and post-processing.

## Current boundary

This block is a streamed compute tile, not a complete accelerator. DDR tile
loading now has a packed-W4 wide-reader front end, but arbitration and connection
to the system memory driver remain top-level integration work. Matched
BRAM-backed ping-pong operand buffers are implemented, while output writeback,
requantization, attention
reductions, normalization, activation functions, command scheduling, and
performance counters remain integration work. The optional W4A8 format should
be preserved through DDR and local weight storage so its bandwidth advantage is
not lost before decode.

The reader requests compact 64-bit qwords from `memory_io_driver`, splits each
word into two 32-bit/8-weight transfers, and holds only one qword locally. It
accepts any positive weight count, reports lane validity on the final vector,
requires a four-byte-aligned base address, and splits large reads at the cache's
64-byte line boundary.
