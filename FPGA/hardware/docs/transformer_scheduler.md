# Transformer GEMM scheduler and DDR top

`bcuda_transformer_ddr_top` is the first deployable top for the Transformer
migration. It connects AXI-Lite control, the cached DDR memory driver, debug
counters, and one 8x8 DSP GEMM array. It supports W8A8 and packed W4A8 weights,
arbitrary positive M/N edge sizes, K up to 1024, and row-major INT32 output.

## DDR layouts

Inputs are tile-major so the PL can consume one complete eight-lane vector
without a transpose or gather unit:

- A: `[ceil(M/8)][K][8]` signed INT8 bytes. The innermost lane is an M row.
- B W8: `[ceil(N/8)][K][8]` signed INT8 bytes. The innermost lane is an N column.
- B W4: `[ceil(N/8)][K][4]`; lane 0 is the low nibble, lane 1 the high nibble,
  with two's-complement values in the range -8 through 7.
- C: ordinary row-major `[M][N]` signed INT32, using `C_STRIDE` bytes per row.

Unused lanes in the final A or B tile must be zero. The scheduler masks invalid
C rows and columns. Python helpers `pack_transformer_activations_int8` and
`pack_transformer_weights` implement these layouts.

## Launch registers

The existing AXI-Lite offsets are retained. Program A/B/C base addresses,
`C_STRIDE`, M, N, K, tile size 8, and array count 1. Bit 0 of CONTROL starts an
operation; bit 11 selects packed W4 weights. STATUS bit 0 is busy, bit 1 is
done, and bit 2 is error. Debug and 64-bit cycle counters occupy 0x80-0xc8.

The scheduler loads one A tile and one B tile into BRAM, prefetches synchronous
BRAM outputs, runs K vectors through the shared array, and writes valid C
elements to DDR. It retains each A tile while walking every N tile, eliminating
the former repeated activation reads. W8 loads use cache-line-sized requests
and stream up to eight vectors per request. The synchronous BRAM feed is
pipelined after its first read, sustaining one K-vector per clock.

Control bit 12 selects the fused INT32-to-tile-major-INT8 requantize-pack path.
This removes the row-major-to-tile-major host copy between consecutive linear
layers. The graph runtime records layouts 5/6/7 for tiled activations, W8
weights, and W4 weights respectively.

## Verified checkpoint

The scheduler passes direct W8/W4 golden tests and an integrated DDR-top golden
test. The complete PS/PL implementation meets 100 MHz with +0.229 ns setup
slack and +0.044 ns hold slack and uses 68 of 220 DSP48E1s. Vivado emits `.bit`,
`.hwh`, `.bin`, and `.xsa` board artifacts under `FPGA/hardware`.

This top is a GEMM execution engine, not yet a complete DeiT encoder engine.
LayerNorm, GELU, Softmax, bias, and residual blocks have standalone RTL tests
but still need DDR-stream wrappers and top-level operation scheduling.
Until then, the firmware graph executor supplies CPU fallbacks for these ops,
allowing hybrid MLP-block execution around the accelerated GEMMs.
