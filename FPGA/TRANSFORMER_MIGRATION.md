# Transformer migration boundary

The `DeiT_Tiny` branch deliberately replaces the previously deployed CNN
accelerator. The CNN work is already preserved remotely, so this branch keeps
only infrastructure that has direct value for the Transformer implementation.

## Retained

- HP0 AXI memory transport, PYNQ board integration, CMA allocation, socket
  protocol, and buffer management
- INT8 DSP MAC and fixed-point requantization primitives
- DeiT-Tiny quantization experiments and the fixed fruit-quality dataset split

## Removed

- CNN training/export scripts, generated models, demo data, and demo entry points
- Conv2D, pooling, CNN post-processing, diagnostics, simulations, and tests
- the 16x16 systolic core, old tiled controller, and array-cluster prototype
- old bitstreams, routed netlists, reports, and CNN-focused documentation

## Temporarily retained for refactoring

Some compiler, graph, firmware, AXI-control, and Vivado integration files still
contain parts of the old command/register contract. They also contain reusable
transport logic, so they will be refactored against the Transformer command set
rather than deleted blindly. They do not currently define a deployable design.

Tracked deletions are recoverable from Git history. Locked ignored Python cache
files may remain under `FPGA/train_model`; they are not source or build inputs.

## First hardware checkpoint

The migration now has a synthesized 8x8 matrix tile. It uses exactly 64 DSP48E1
blocks and shares that datapath between W8A8 and weight-only W4A8. Packed INT4
weights are sign-extended only at array ingress, preserving their storage and
transfer advantage without maintaining a second arithmetic array. W4A8
ping-pong buffers, LayerNorm, GELU, Softmax, bias/residual arithmetic,
cycle/debug counters, firmware fallbacks, and a Python DeiT MLP graph builder
are now implemented. A tiled DDR GEMM scheduler and production top support W8A8
and packed W4A8, pass integrated simulation, and pass complete PYNQ-Z2
place-and-route at 100 MHz. The graph ABI carries the physical activation and
weight layouts, firmware routes tiled GEMMs and fused requantize-pack to the PL,
and the Python builder emits working W8 or W4 DeiT MLP blocks. The complete
QKV/head scheduler remains the next architectural boundary.
