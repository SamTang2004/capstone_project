# PYNQ DeiT accelerator

This branch develops a Transformer-oriented accelerator for DeiT-Tiny fruit
quality classification on PYNQ-Z2. The deployment precision is W4A8: packed
INT4 matrix weights, INT8 activations and attention probabilities, and wider
accumulation.

The previous CNN implementation is preserved in Git history and has been
removed from this branch so the new architecture can be reviewed without two
competing datapaths.

- `testing/deit_tiny`: training, quantization, and measured accuracy experiments
- `FPGA/hardware`: reusable PYNQ memory infrastructure and new RTL
- `FPGA/firmware`: PS-side buffer, protocol, and accelerator control runtime
- `FPGA/budgetcuda`: host-side transport and tensor utilities

The new 8x8 Transformer GEMM top passes simulation and complete 100 MHz PYNQ-Z2
implementation. The host compiler and firmware can now execute a pre-norm DeiT
MLP as a hybrid graph: tiled W8A8/W4A8 GEMMs and fused requantize-pack use the
PL, while LayerNorm, GELU, and residual addition retain deterministic firmware
fallbacks. Full attention scheduling and board-level numerical validation remain
before end-to-end DeiT image inference.
