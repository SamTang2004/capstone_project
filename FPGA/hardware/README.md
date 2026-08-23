# Transformer accelerator hardware

The active target is a PYNQ-Z2 Transformer accelerator built around one
pure-DSP 8x8 INT8 matrix array. A second array is considered only after measured
memory utilization shows that it can be kept busy.

## Preserved infrastructure

- 64-bit PS HP0 AXI3 bridge and burst-capable memory driver
- memory-I/O stress tests and INT8 DSP MAC primitive
- fixed-point requantization building blocks
- PYNQ board files, constraints, and integration scaffolding

## New implementation boundary

The first compute block is now implemented: an 8x8, 64-DSP GEMM tile with INT8
activations, selectable packed-INT4 or INT8 weights, INT32 accumulation, edge
masks, and a valid/ready streaming interface. Both W8A8 and W4A8 modes pass
deterministic RTL golden tests. See [the GEMM tile contract](docs/gemm_8x8_w4a8.md).

The next integration layer now has ping-pong operand buffers, fused bias and
residual arithmetic, LayerNorm, GELU, attention Softmax, and cycle/debug
counters. Each block has a deterministic RTL test and an out-of-context
synthesis report. See [the primitive and debug contract](docs/transformer_primitives.md).

The first [DDR GEMM scheduler and integrated top](docs/transformer_scheduler.md)
now connects AXI-Lite control, the memory path, debug counters, and the 8x8
array. W8A8/W4A8 scheduler and integrated-top simulations pass. The complete
PYNQ-Z2 image is routed at 100 MHz with +0.213 ns setup and +0.060 ns hold
slack. Full QKV/attention/MLP scheduling remains beyond this GEMM checkpoint.

Packed W4 ingress now includes a [wide weight reader](rtl/packed_w4_weight_reader.sv).
It consumes the retained 64-bit cached qword stream, emits two 8-weight vectors
per qword, splits requests at 64-byte cache-line boundaries, and needs only one
64-bit local holding entry.

The [W4A8 operand-buffer pair](docs/operand_buffers.md) adds two BRAM-backed
activation banks and two packed-weight banks. It supports concurrent fill/drain,
independent ingress backpressure, and lockstep delivery to the GEMM tile.

The production source manifest and PS/PL wrapper now target
`bcuda_transformer_ddr_ip`. Board numerical and endurance tests remain required
before treating the generated image as a release bitstream.
