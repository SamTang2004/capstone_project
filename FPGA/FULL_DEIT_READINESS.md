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
- A streamed manifest runner covering image preprocessing, patch embedding,
  class/position tokens, all 12 encoder blocks, final LayerNorm, and the head
- CUDA-style resident model storage: 417 W8 model-constant buffers totaling
  5,599,352 bytes remain in DDR3 and are borrowed by each graph invocation;
  this includes class and position embeddings

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
- A complete W8A8 image run on the PYNQ-Z2 classified the fresh-apple smoke
  image as `apple_fresh`, with logits `[3, -2, 1, 1, -1, -6]`.
- The corrected complete-image host timing was 45.73 seconds, including a
  0.46-second patch-embedding stage. The path repeatedly allocates, uploads,
  executes, reads, and frees 27 graphs; it is a functional-integration path,
  not a resident-model kernel benchmark.
- With matrix and normalization constants resident, one-time DDR loading took 13.31 seconds. Two
  warm physical runs took 29.13 and 29.53 seconds (29.33-second mean), retained
  the same `apple_fresh` logits, and improved latency by 35.3% (1.54x) versus
  the 45.31-second streamed-weight baseline.
- Moving the class and position embeddings into the resident set preserved the
  logits. The final strict device-owned model loaded in 12.28 seconds and ran
  in 28.48 seconds; its token-embedding accelerator stage took 0.14 seconds.
- Peak persistent constants plus the largest transient attention workspace and
  descriptor occupy approximately 7.60 MB of the 16 MiB arena. The firmware
  allocation table now has 512 slots; the final resident configuration uses
  417, with at most 75 graph buffers in one stage.

## Remaining deployment requirements

The complete-image path is functionally integrated, but its 28.48-second warm
host time does not meet the less-than-500-ms objective. Weights are now
persistent; the next step is to retain reusable activation workspaces and all
28 descriptors, then execute the schedule through one command instead of 28
host round trips. Adjacent scheduling steps can subsequently be fused and
hardware-only counters profiled separately from transport time. End-to-end
board accuracy must also be measured over the held-out split; one correctly
classified smoke image is not an accuracy result.
