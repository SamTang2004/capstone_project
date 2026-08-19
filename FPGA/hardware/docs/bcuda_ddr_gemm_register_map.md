# Budget-CUDA Tiled Systolic DDR Register Map

The final standalone PL top is `bcuda_tiled_systolic_ddr_top`. It exposes 16
writable control registers and 16 read-only Conv2D performance registers. The PYNQ-Z2 PS/PL wrapper maps this
window at `0x43C80000`. This avoids the PYNQ base overlay's occupied
`0x43C00000` through `0x43C70000` control range.

| Offset | Name | Access | Description |
| --- | --- | --- | --- |
| `0x00` | `CONTROL` | RW | Bit 0 starts an operation. Bit 1 selects `REQUANTIZE`; clear selects tiled GEMM. Firmware writes `0`, programs operands, then writes the operation word. |
| `0x04` | `STATUS` | R | Bit 0 busy, bit 1 done, bit 2 error. Firmware clears `CONTROL` after done/error. |
| `0x08` | `A_BASE` | RW | 32-bit byte address of INT8 A tile in row-major DDR memory. |
| `0x0c` | `B_BASE` | RW | 32-bit byte address of INT8 B tile in row-major DDR memory. |
| `0x10` | `C_BASE` | RW | 32-bit byte address of INT32 C tile output in row-major DDR memory. |
| `0x14` | `A_STRIDE` | RW | A row stride in bytes. For a dense 16x16 INT8 tile this is `16`. |
| `0x18` | `B_STRIDE` | RW | B row stride in bytes. For a dense 16x16 INT8 tile this is `16`. |
| `0x1c` | `C_STRIDE` | RW | C row stride in bytes. For a dense 16x16 INT32 tile this is `64`. |
| `0x20` | `M` | RW | GEMM M dimension. |
| `0x24` | `N` | RW | GEMM N dimension. |
| `0x28` | `K` | RW | GEMM K dimension. |
| `0x2c` | `TILE_SIZE` | RW | Tile size. Current firmware programs `16`. |
| `0x30` | `ARRAY_COUNT` | RW | Number of active 16x16 systolic arrays. The PYNQ-Z2 deployment bitstream implements one array, so firmware must program `1`. The source-level cluster remains parameterizable up to eight arrays for a larger FPGA target. |
| `0x34` | `RQ_MULTIPLIER` | RW | Signed Q-format scalar multiplier for `REQUANTIZE`. |
| `0x38` | `RQ_RIGHT_SHIFT` | RW | Arithmetic right shift, `0..31`, for `REQUANTIZE`. |
| `0x3c` | `RQ_OUTPUT_ZERO_POINT` | RW | Signed output zero point for `REQUANTIZE`. |

Offsets `0x40` through `0x7c` are read-only counters for the most recently
completed Conv2D command: total, zero-A, load-A, zero-B, load-B, bias, compute,
accumulate, write, loader-address, loader-request, loader-receive, cache hits,
cache misses, AXI line-fill cycles, and bytes delivered, in that order.

`REQUANTIZE` uses `A_BASE` as aligned INT32 input, `C_BASE` as aligned INT8 output, and `M` as the contiguous element count. Its PL formula is `sat_int8(((x * RQ_MULTIPLIER) >>> RQ_RIGHT_SHIFT) + RQ_OUTPUT_ZERO_POINT)`. The output writer packs four INT8 values into each DDR write; the final partial word uses byte strobes. This MVP PL path requires symmetric INT32 accumulators (`input_zero_point = 0`).

Current constraints:

- The hardware controller is tiled; the current firmware launch path still
  programs one 16x16 tile per GEMM operation while model-deployment scheduling
  matures.
- A and B must be INT8; C must be INT32.
- Physical addresses programmed into the PL top must fit in 32 bits.
- The checked-in PYNQ-Z2 configuration uses one array because an eight-array
  16x16 implementation does not fit the Zynq-7020 fabric. The eight-array
  cluster and scheduling tests are retained as an architectural prototype.
