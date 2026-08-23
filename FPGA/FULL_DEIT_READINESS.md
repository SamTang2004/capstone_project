# Full DeiT FPGA readiness

## Ready now

- One 8x8 INT8 DSP array with W8A8 and packed W4A8 weight modes
- Tile-major activation and weight packers in Python, firmware, and the graph ABI
- Tiled DDR GEMM scheduler for arbitrary M/N edges and K up to 1024
- Fused row-major INT32 requantization to tile-major INT8 activations
- LayerNorm, GELU, Softmax, residual, bias, counters, and debug RTL unit tests
- Hybrid pre-norm DeiT MLP graph with W8 or W4 weights
- Dynamic row-major/tiled activation transpose and A/B packing in RTL, firmware, host ABI, and mock runtime
- Complete three-head DeiT attention schedule, including Q/K/V, QK^T, Softmax, AV, concatenation, projection, and residual
- DDR-backed LayerNorm, Softmax, GELU, and residual engines
- Six-class DeiT-Tiny checkpoint exporter for patch embedding, tokens, 12 encoder blocks, final norm, and head
- W8A8 conformance package: 74 tensors, 5,457,408 packed bytes, and zero sampled integer-GEMM accumulator error
- Integrated legacy row-major INT32-to-INT8 requantization, with a sentinel
  overrun guard in the top-level RTL test
- 100 MHz PYNQ-Z2 routed image: WNS +0.008 ns, WHS +0.058 ns, 88/220 DSPs,
  10,520/53,200 LUTs, and 4/140 BRAM tiles
- Physical W8A8/W4A8 GEMM, fused requantize-pack, MLP, and complete three-head
  attention validation on the PYNQ-Z2

## Completed encoder requirements

1. Dynamic activation transpose/pack operation: complete and RTL-tested.
2. Complete attention sequence scheduler: complete and host/mock-tested.
3. DDR-backed non-GEMM RTL: complete and RTL/firmware-tested.
4. Full six-class DeiT-Tiny exporter: complete; W8A8 is the deployment default
   and W4A8 remains an explicit research option.
5. Local numerical conformance and held-out accuracy: complete. Shared-scale
   W8A8 reached 98.88% accuracy versus 99.44% for the source checkpoint. The
   shared-scale W4A8 experiment reached only 21.91%, confirming that the current
   scalar-requantization hardware should deploy W8 weights.

## Physical validation completed

- The prior HP0 AXI B-channel deadlock is resolved.
- A 100-W8 plus 100-W4 endurance run completed 201 operations (including the
  final pack) with no latched error.
- W8A8 and W4A8 graph runs each completed the MLP and complete three-head
  attention schedules on physical hardware.
- The graph-level row-major requantization path is now integrated in the
  Transformer top. Its previous omission allowed control bit 1 to fall through
  to a stale GEMM command and overwrite neighboring buffers; the corrected
  image passes both RTL overrun checks and physical residual-add execution.
- Debug event inputs are registered at the control/status boundary. This keeps
  trace collection off the shared-DDR handshake critical path and restored
  100 MHz timing closure.

## Remaining full-run requirement

The remaining gap is an executable complete-image package and runner that joins
patch extraction/embedding, class and position tokens, all 12 encoder blocks,
final LayerNorm, and the six-class head. The exporter contains all 74 tensors,
but physical full-image latency and end-to-end board accuracy cannot be measured
until that orchestration package is generated and run. Kernel and encoder-graph
hardware validation should not be reported as complete-image DeiT inference.
