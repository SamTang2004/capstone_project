# Packed INT4 Tensor ABI

Status: active packed-INT4 deployment ABI.

## Scalar Format

- Values are signed two's-complement INT4 in the range `[-8, 7]`.
- Even logical columns occupy byte bits `[3:0]`.
- Odd logical columns occupy byte bits `[7:4]`.
- Hardware sign-extends each selected nibble before multiplication.
- Bias and accumulation remain signed INT32.

## Row Layout

Each matrix or tensor row starts on a byte boundary:

```text
byte 0: column 0 in low nibble, column 1 in high nibble
byte 1: column 2 in low nibble, column 3 in high nibble
...
```

For an odd row width, the unused high nibble of the final byte is zero.
`row_stride_bytes` is measured in bytes and must be at least
`ceil(logical_columns / 2)`.

Row alignment deliberately avoids nibble-level read-modify-write operations at
tile boundaries. A later packed-contiguous tensor ABI may add a starting-nibble
field if its memory savings justify the additional address and tail logic.

## Generic Tile Interfaces

`tile_loader_int4_ddr`:

- supports tiles up to 8x8;
- reads one packed byte for two adjacent elements;
- emits one signed nibble cache write per logical element;
- produces 32 DDR reads for a full 8x8 tile.

`tile_writer_int4_ddr`:

- accepts already-requantized signed INT4 tile values;
- reads up to two values and coalesces them into one byte write;
- zero-fills an unused odd-width high nibble;
- produces 32 DDR writes for a full 8x8 tile.

`gemm_dual_8x8_int4_core`:

- stores pre-skewed operand waves in four small distributed memories;
- executes two 8x8 arrays with INT32 accumulators;
- supports independent, shared-A, and shared-B modes;
- uses no DSP blocks in the current LUT-only configuration.

## Reference Policy

The current INT8 deployment files remain unchanged until the INT4 burst loader,
requantizer, controller, top, firmware ABI, and host packing pass end-to-end
simulation:

- `tile_loader_ddr.sv`
- `tile_writer_ddr.sv`
- `conv2d_tile_loader_ddr.sv`
- `conv2d_tile_writer_ddr.sv`
- `pe_int8_mac.sv`
- `systolic_array_16x16.sv`
- `gemm_16x16_core.sv`
- `gemm_array_cluster_8.sv`

The complete presentation-era implementation is also preserved by Git branch
`Checkpoint-1` at commit `ae8fcb3`. Only after a deployable INT4 top passes
virtual and board validation should the in-tree INT8 files move to
`FPGA/deprecated`.

## Implemented Migration

- The burst Conv2D and matrix loaders unpack packed INT4 DDR traffic.
- Requantization writes packed INT4 with truncation and saturation.
- Activation, pooling, upsample, and concat use packed INT4 interfaces.
- GEMM scheduling targets two 8x8 arrays.
- `bcuda_dual_int4_ddr_top` implements the active register and DDR path.
- Firmware, host allocation, ONNX lowering, QAT export, and the mock device use
  the shared packed format.
