# Transformer GEMM scheduler and DDR top

`bcuda_transformer_ddr_top` is the first deployable top for the Transformer
migration. It connects AXI-Lite control, the cached DDR memory driver, debug
counters, and two independent 8x8 DSP GEMM arrays. It supports W8A8 and packed
W4A8 weights, arbitrary positive M/N edge sizes, K up to 1024, and row-major
INT32 output.

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

The scheduler loads up to two independent A tiles and one shared B tile into
BRAM. For M dimensions larger than eight, it assigns adjacent 8-row M tiles to
the two cores and broadcasts each B vector to both, producing up to 16 rows in
parallel without reading the weight tile twice. Either core has its own start,
feed, completion, and result state; the final unpaired M tile runs only on core
0. The scheduler retains both A tiles while walking every N tile, eliminating
repeated activation reads. W8 loads use cache-line-sized requests and stream up
to eight vectors per request. The synchronous BRAM feeds are pipelined after
their first read and use a lock-step issue handshake whenever both cores are
active, sustaining one K-vector per clock without allowing their K indices to
diverge.

Control bit 12 selects the fused INT32-to-tile-major-INT8 requantize-pack path.
This removes the row-major-to-tile-major host copy between consecutive linear
layers. The graph runtime records layouts 5/6/7 for tiled activations, W8
weights, and W4 weights respectively.

LayerNorm and Softmax share a dedicated rowwise engine. It loads one row into
local storage, computes one Q32 reciprocal per row with the iterative divider,
and replaces subsequent element divisions with reciprocal multiplies and exact
quotient correction. LayerNorm retains integer variance/square-root, Q14
normalization, gamma/beta, and saturation. Softmax retains the Q15 exponential
lookup and Q7 rounded output. The DDR command and firmware interfaces are
unchanged.

## Verified checkpoint

The scheduler passes direct W8/W4 golden tests, a dual-core asymmetric-edge
test (M=10), and an integrated DDR-top golden test. The rowwise engine passes
standalone tests and model-dimension bit-exact differential tests against the
legacy units (192-element LayerNorm and 197-element Softmax). The complete
PS/PL implementation meets 100 MHz with +0.070 ns setup slack and +0.025 ns
hold slack. It uses 158 of 220 DSP48E1s, 7.5 BRAM tiles, 11,432 slice LUTs, and
11,373 slice registers. Vivado emits `.bit`, `.hwh`, `.bin`, and `.xsa` board
artifacts under `FPGA/hardware`.

The top executes one graph operation at a time; firmware remains responsible
for issuing the full DeiT graph. LayerNorm, GELU, Softmax, residual addition,
requantization, and activation packing have DDR-backed hardware paths. The two
GEMM cores are scheduled independently at tile granularity inside a launch,
while data-dependent graph operations remain sequential across launches.
