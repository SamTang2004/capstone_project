# W4A8 operand buffers

`w4a8_operand_buffers` places matched ping-pong buffers between memory ingress
and the 8x8 GEMM tile. The activation side stores one 64-bit vector per K-step;
the weight side stores one 32-bit vector containing eight packed W4 weights.

## Organization

Each side has two synchronous simple-dual-port BRAM banks. One bank can fill
while the other drains. The default depth is 1024 K-steps per bank, enough for
the 768-element hidden dimension in the DeiT-Tiny MLP. Depth is parameterized so
future models can trade BRAM capacity against K tiling.

At the default depth, storage is:

- activation banks: 2 x 1024 x 64 bits = 16 KiB
- packed-weight banks: 2 x 1024 x 32 bits = 8 KiB
- total payload: 24 KiB in six RAMB36E1 tiles

The two write streams are independently backpressured. A tile becomes available
for draining only after both streams have accepted exactly `k_steps` entries.
The read side pairs both synchronous BRAM outputs behind one valid/ready
interface. Neither side advances unless its partner is valid, preventing an
activation/weight mismatch under compute backpressure.

## Pipeline behavior

1. Assert `fill_start` with `k_steps` when `fill_start_ready` is high.
2. Fill activation and packed-weight streams independently.
3. `tile_available` rises when both corresponding banks are complete.
4. Assert `drain_start` when `drain_start_ready` is high.
5. After one synchronous-BRAM read cycle, consume one paired K-step per cycle.
6. `operand_last` marks the final pair and `drain_done` releases both banks.

The second bank may fill throughout steps 4-6. The current paired wrapper drains
each tile once. Cross-output-tile activation reuse and cross-row-tile weight
reuse belong in the scheduler or a future retained-bank mode.

## Synthesis checkpoint

Out-of-context synthesis for `xc7z020clg400-1` at 100 MHz reports 187 LUTs,
156 registers, six block-RAM tiles, zero DSPs, and 4.223 ns setup slack. This is
the buffer pair alone, excluding the memory reader and GEMM array.
