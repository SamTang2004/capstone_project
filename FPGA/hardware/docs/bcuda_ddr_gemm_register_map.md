# Budget-CUDA Tiled Systolic DDR Register Map

The final standalone PL top is `bcuda_dual_int4_ddr_top`, exposed through
`bcuda_tiled_systolic_ddr_ip`. It has 16 control registers followed by 16
read-only diagnostic registers.
The PYNQ-Z2 PS/PL wrapper maps this
window at `0x43C80000`. This avoids the PYNQ base overlay's occupied
`0x43C00000` through `0x43C70000` control range.

| Offset | Name | Access | Description |
| --- | --- | --- | --- |
| `0x00` | `CONTROL` | RW | Bit 0 START; bit 1 REQUANTIZE; bit 2 CONV2D; bit 3 ACTIVATION; bit 4 LEAKY; bit 5 MAXPOOL; bit 6 UPSAMPLE; bit 7 CONCAT; bit 8 INT4; bit 9 AVGPOOL; bit 10 fused Conv output; bit 11 fusion-config command; bit 12 fused activation enable. Exactly one operation-select bit may be set. Bits 4, 10, and 12 are modifiers. Plain INT4 GEMM uses START+INT4. |
| `0x04` | `STATUS` | R | Bit 0 busy, bit 1 done, bit 2 error. Firmware clears `CONTROL` after done/error. |
| `0x08` | `A_BASE` | RW | Primary packed INT4 input, or aligned INT32 requantization input. |
| `0x0c` | `B_BASE` | RW | Packed INT4 GEMM/Conv weights or concat input 1. |
| `0x10` | `C_BASE` | RW | INT32 GEMM/Conv output or word-aligned packed INT4 operator output. |
| `0x14` | `A_STRIDE` | RW | GEMM A row bytes; tensor-op N otherwise. |
| `0x18` | `B_STRIDE` | RW | GEMM B row bytes; tensor-op channel count otherwise. |
| `0x1c` | `C_STRIDE` | RW | GEMM C row bytes; tensor-op input height otherwise. |
| `0x20` | `M` | RW | GEMM M dimension. |
| `0x24` | `N` | RW | GEMM N dimension. |
| `0x28` | `K` | RW | GEMM K dimension. |
| `0x2c` | `TILE_SIZE` | RW | INT4 tile size (`8`) or pooling/Conv kernel height. |
| `0x30` | `ARRAY_COUNT` | RW | INT4 array count (`2`) or pooling/Conv kernel width. |
| `0x34` | `RQ_MULTIPLIER` | RW | Requant multiplier; tensor-op stride pair or concat C1. |
| `0x38` | `RQ_RIGHT_SHIFT` | RW | Requant/Leaky shift; tensor-op padding pair or concat C2. |
| `0x3c` | `RQ_OUTPUT_ZERO_POINT` | RW | Requant zero point, Conv bias base, or concat input 2 base. |
| `0x40` | `DIAG_TOTAL_CYCLES` | R | Command-active cycles. |
| `0x44` | `DIAG_ENGINE_CYCLES` | R | Tiled compute engine busy cycles. |
| `0x48` | `DIAG_COMPUTE_CYCLES` | R | Systolic core active cycles. |
| `0x4c` | `DIAG_LOAD_CYCLES` | R | Engine cycles waiting for a tile group. |
| `0x50` | `DIAG_CACHE_WAIT_CYCLES` | R | Load-wait cycles with no ready look-ahead tile. |
| `0x54` | `DIAG_CACHE_READY_CYCLES` | R | Cycles with at least one ready cached tile. |
| `0x58` | `DIAG_DDR_READ_COMMANDS` | R | Accepted external read commands. |
| `0x5c` | `DIAG_DDR_READ_BEATS` | R | Accepted external read response beats. |
| `0x60` | `DIAG_DDR_WRITE_COMMANDS` | R | Accepted external write commands. |
| `0x64` | `DIAG_DDR_STALL_CYCLES` | R | Cycles with blocked command/read/write handshakes. |
| `0x68` | `DIAG_CORE_ROWS` | R | Tile rows delivered to the compute engine. |
| `0x6c` | `DIAG_OUTPUT_STALL_CYCLES` | R | Result-valid cycles blocked by the writer. |
| `0x70` | `DIAG_DESCRIPTOR_COUNT` | R | Accepted look-ahead tile descriptors. |
| `0x74` | `DIAG_CACHE_FULL_CYCLES` | R | Cycles where look-ahead allocation was full. |
| `0x78` | `DIAG_CACHE_OCCUPANCY` | R | Bytes: max allocated, max ready, final allocated, final ready. |
| `0x7c` | `DIAG_MAGIC` | R | Diagnostics ABI marker `0x44494132` (`DIA2`). |

The counters reset on every accepted command dispatch and freeze at command
completion. Firmware snapshots them before launching the next layer.

`REQUANTIZE` uses `A_BASE` as aligned INT32 input, `C_BASE` as
word-aligned packed INT4 output, and `M` as the logical element count. Its PL
formula is
`sat_int4(((x * RQ_MULTIPLIER) >>> RQ_RIGHT_SHIFT) + RQ_OUTPUT_ZERO_POINT)`.
The final odd high nibble is zero and partial words use byte strobes.

For a fused `Conv2D -> Requantize -> Activation` chain, firmware first issues
`START|INT4|FUSION_CONFIG`. During that command, `RQ_MULTIPLIER` is the signed
multiplier, `RQ_RIGHT_SHIFT[5:0]` is the requant shift,
`RQ_RIGHT_SHIFT[10:8]` is the optional LeakyReLU shift, and
`RQ_OUTPUT_ZERO_POINT` is the signed output zero point. It then issues the
normal Conv command with bit 10 set. The result writer retains one 8x16 output
tile, requantizes and activates it, and writes packed INT4 directly. Firmware
uses this path only when the NCHW channel plane is divisible by eight logical
elements; other shapes retain the separate operator path.

Current constraints:

- GEMM and Conv inputs are signed two's-complement packed INT4 with the even
  logical element in the low nibble; accumulators and their output are INT32.
- Requantization, activation, pooling, upsample, and concat outputs are packed
  INT4 and require a word-aligned output base.
- Physical addresses programmed into the PL top must fit in 32 bits.
- The memory driver accepts eight ordered read requests. The Conv loader holds
  a 64-element tagged reorder window and retires returned bytes in tile order.
- The production compute path uses two LUT-only 8x8 arrays. Legacy INT8 RTL is
  excluded from the final bitstream manifest.
