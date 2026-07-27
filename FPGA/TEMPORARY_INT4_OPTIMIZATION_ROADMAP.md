# Temporary INT4 Optimization Roadmap

Status: architecture exploration
Target: PYNQ-Z2 / XC7Z020, 100 MHz minimum
Baseline: Checkpoint-1, one mixed DSP/LUT 16x16 INT8 array

## Current INT4 Implementation Baseline

The production candidate now uses two LUT-only 8x8 signed INT4 arrays. DSP use
is permitted outside the arrays for pipelined address generation and other
control/datapath arithmetic where synthesis shows a timing benefit. A separate
DSP budget remains available for experimental packed-product or hybrid array
variants; those variants must not silently change the production array.

The July 23 out-of-context synthesis of `bcuda_dual_int4_ddr_top` for the
XC7Z020 at 100 MHz reports:

- WNS: `+0.396 ns`
- LUTs: `11,193`
- FFs: `10,431`
- RAMB18: `4`
- DSP48E1: `13` of `220`
- DSP48E1 inside both systolic arrays: `0`
- Unallocated DSP48E1 slices: `207`

The 13 allocated DSPs are currently used by non-array arithmetic, including
the pipelined Conv2D address path. This resource split is intentional: spend
DSPs where they remove control or address-generation timing pressure, while
retaining enough headroom to compare experimental DSP-backed INT4 processing
elements against the LUT baseline.

## 1. Corrected Architecture Conclusions

The proposed direction is reasonable:

- Replace the monolithic 16x16 array with two independently schedulable 8x8
  arrays.
- Make signed INT4 the primary activation and weight format.
- Keep INT32 bias and accumulation until model-level range analysis proves that
  a narrower accumulator is safe.
- Add activation or weight sharing between the two arrays.
- Prefer a minimally buffered streaming path for low-reuse operations and an
  optional BRAM reuse path for GEMM and convolution.
- Defer FP4 and sparse execution until the dense INT4 design is measured.

One correction is essential: DSP48E1 SIMD mode does not provide SIMD
multiplication. `TWO24` and `FOUR12` split only the 48-bit
adder/subtractor/accumulator. AMD requires the DSP multiplier to be disabled in
those modes. Therefore, the candidate compute implementations are:

1. LUT-only signed INT4 multipliers with fabric or BRAM-backed accumulation.
2. One conventional INT4 multiply per DSP48E1 in `ONE48` mode.
3. Packed multiplication in which one shared operand is multiplied by multiple
   packed independent operands, with guard bits and signed correction logic.
4. A measured hybrid of LUT and packed-DSP processing elements.

Packed multiplication is not the DSP48E1 SIMD mode. It must be proven
bit-exact, synthesized, and compared with the LUT implementation before it
becomes the production architecture.

References:

- AMD DSP48E1 primitive:
  https://docs.amd.com/r/en-US/ug953-vivado-7series-libraries/DSP48E1
- AMD 7 Series DSP48E1 Slice User Guide:
  https://docs.amd.com/api/khub/documents/gu4oRPFEh_Pm2uaAlfY6Kg/content
- AMD Zynq-7000 DSP description:
  https://docs.amd.com/r/en-US/ug585-zynq-7000-SoC-TRM/Digital-Signal-Processing-DSP-Slice

## 2. First-Order Throughput and Bandwidth

At 100 MHz, two 8x8 arrays contain 128 processing elements:

```text
peak MAC rate = 2 arrays * 8 * 8 * 100 MHz
              = 12.8 GMAC/s
peak operation rate, counting multiply and add separately
              = 25.6 GOPS
```

For one K step, an independent 8x8 array consumes eight A values and eight B
values. With packed INT4, two independent arrays require:

```text
2 arrays * (8 A + 8 B) * 4 bits = 128 bits/cycle
100 MHz * 16 bytes/cycle        = 1.6 GB/s
```

When both arrays compute the same M tile and adjacent N tiles, they share the A
activation vector:

```text
(8 shared A + 16 independent B) * 4 bits = 96 bits/cycle
100 MHz * 12 bytes/cycle                 = 1.2 GB/s
```

This is a 25% reduction in operand traffic compared with two independent
arrays. The symmetric schedule shares B weights while the arrays compute two M
tiles.

The active BudgetCUDA AXI bridge is currently 32 bits wide and permits one
outstanding transaction. Its absolute payload ceiling at 100 MHz is only:

```text
32 bits * 100 MHz = 0.4 GB/s
```

A full 64-bit HP port at the same clock has a raw ceiling of 0.8 GB/s before
protocol gaps and DDR contention. Consequently, neither an independent
1.6 GB/s feed nor a shared 1.2 GB/s feed can be sustained from one 100 MHz HP
port without reuse. This is the immediate architectural constraint.

Two 64-bit HP paths have a combined raw ceiling of 1.6 GB/s at 100 MHz, but
real sustained bandwidth will be lower. Optional on-chip reuse, wider or faster
AXI clocks, multiple HP ports, and enough outstanding bursts must be evaluated
together. Output traffic and refresh/contention further reduce the available
read bandwidth.

For the shared-A schedule, each K step performs 128 MACs from 12 input bytes:

```text
operand arithmetic intensity = 128 / 12 = 10.67 MAC/byte
```

Without sharing, it is 8 MAC/byte. These figures exclude amortized output
writes and any repeated loads caused by small buffers.

## 3. Dual-Array Sharing Modes

The array pair should expose three scheduling modes:

| Mode | Array 0 | Array 1 | Shared stream | Intended use |
|---|---|---|---|---|
| Independent | Arbitrary tile | Arbitrary tile | None | Irregular tails and unrelated work |
| Shared A | `C[M0,N0]` | `C[M0,N1]` | A activations | Conv/GEMM output-channel parallelism |
| Shared B | `C[M0,N0]` | `C[M1,N0]` | B weights | Batch/spatial-row parallelism |

Sharing must occur before duplicated FIFOs and unpackers. Otherwise, it saves
DDR traffic but needlessly duplicates internal switching and storage.

Each array retains independent valid, ready, clear, accumulator, and completion
state. One stalled or tail-masked array must not corrupt the other array's
wavefront.

## 4. INT4 Numerical Contract

Define the data format before changing RTL:

- Activations and weights are signed two's-complement INT4 in `[-8, 7]`.
- Two values are packed per byte; the nibble order is fixed in the ABI.
- Odd tensor tails use a zero-filled unused nibble and an explicit logical
  element count.
- Multiply results are sign-extended before accumulation.
- Bias and accumulation remain signed INT32.
- Requantization uses the existing truncation policy, then clamps to INT4.
- Zero padding uses the quantized representation of real zero. The first
  implementation should retain symmetric zero points so that this value is
  nibble `0`.
- Per-channel weight scales remain a host/importer and requantization concern;
  they do not change the array multiplier.

INT8 model import may remain as a conversion input, but the deployed graph and
PL memory ABI should be explicitly versioned for packed INT4. Silent mixing of
INT8 and INT4 buffers is not allowed.

FP4 is deferred because it needs a selected encoding, exception and subnormal
policy, scaling rules, conversion hardware, and an accumulation format. It is
not a drop-in replacement for the signed INT4 datapath.

## 5. Implementation Phases

### Phase 0: Preserve and Instrument the Baseline

- Keep `Checkpoint-1` immutable as the July 28 INT8 reference.
- Record latency, layer utilization, HP0 payload bandwidth, burst efficiency,
  stalls, array-active cycles, and total cycles with `budgetcuda.diagnostics`.
- Add counters for bytes requested, bytes transferred, read starvation,
  write backpressure, FIFO occupancy, and useful MAC cycles.

Exit gate: one repeatable board diagnostic report with the current fruit CNN
and a large irregular GEMM.

### Phase 1: INT4 Golden Model and ABI

- Implement Python pack, unpack, saturate, truncate, GEMM, and Conv2D golden
  models.
- Add exhaustive 4-bit multiplier tests for all 256 signed operand pairs.
- Add random and extreme accumulation tests, odd-length tensors, zero padding,
  and irregular M/N/K dimensions.
- Version the host/firmware descriptor and register-map data type fields.

Exit gate: host golden tests define every bit written to and read from DDR.

### Phase 2: Processing-Element Bake-Off

Build isolated, latency-matched implementations:

- `pe_int4_lut`: one signed INT4 MAC per cycle, forced away from DSPs.
- `pe_int4_dsp`: one conventional MAC per DSP48E1 for reference.
- `pe_int4_packed_dsp`: shared-operand packed products with explicit signed
  correction and separate accumulators.

For each implementation, run behavioral simulation, post-synthesis functional
simulation, and out-of-context synthesis at 100, 125, and 150 MHz. Record:

- LUT, FF, DSP, and BRAM usage;
- WNS and achieved Fmax;
- pipeline latency and initiation interval;
- estimated dynamic power;
- placement/routing pressure for a complete 8x8 array.

Exit gate: select the implementation by measured throughput per LUT, throughput
per DSP, timing margin, and power. Do not select packed DSP merely from the
number of nominal INT4 products.

### Phase 3: One 8x8 Array

- Parameterize the existing core for 4-bit operands and an 8x8 wavefront.
- Preserve pipelined operand forwarding and INT32 accumulation.
- Add row/column tail masks.
- Update golden traces and cycle-aware testbenches for the new pipeline.
- Verify K values below, equal to, and above the array dimension.

Exit gate: all golden GEMMs pass with no X propagation and WNS is non-negative
at 100 MHz.

### Phase 4: Dual Arrays and Activation Sharing

- Instantiate two independent 8x8 cores.
- Implement independent, shared-A, and shared-B dispatch modes.
- Broadcast the shared stream once and duplicate only at the array boundary.
- Schedule adjacent N tiles for Conv2D output-channel sharing.
- Fall back to independent mode for tails and small layers.
- Add per-array utilization and shared-byte counters.

Exit gate: shared mode is bit-identical to two independent runs and reports
close to 25% lower operand bytes for balanced A/B tiles.

### Phase 5: DDR Stream Engine

Replace the current 32-bit, single-outstanding bridge:

- Use a 64-bit AXI data path end to end; avoid narrowing to 32 bits internally.
- Support long aligned INCR bursts without returning to idle between beats.
- Permit multiple outstanding read bursts and separate read/write progress.
- Add shallow elastic FIFOs to absorb AXI latency and backpressure.
- Coalesce adjacent INT4 rows and channels into full beats.
- Issue future bursts before the active FIFO reaches its low watermark.
- Evaluate HP0 plus HP1, or a higher AXI clock with a clock-domain FIFO.
- Keep 4 KB AXI boundaries, alignment, tail strobes, IDs, and error responses
  correct.

"Buffer-free" means no full duplicated tile buffer in streaming mode. It does
not mean zero storage: burst assembly, clock crossing, wavefront skew, and
latency hiding require finite FIFOs.

Exit gate: a DDR-only streaming benchmark reports sustained payload bandwidth,
burst occupancy, outstanding depth, and stall causes. No array performance
claim is accepted without these counters.

### Phase 6: Optional BRAM Reuse Mode

- Streaming mode bypasses full tile storage for low-reuse operations.
- Reuse mode stages the operand with the higher reuse in BRAM.
- Shared-A mode retains an A activation tile while B weight tiles advance.
- Shared-B mode retains a B weight tile while A activation tiles advance.
- Bank and partition BRAMs to supply all array-edge lanes each cycle.
- Select the mode per layer from the host cost model using tensor shape,
  expected reuse, transfer bytes, and measured bandwidth.

Exit gate: the same graph executes correctly in both modes, and diagnostics
show that reuse mode increases end-to-end throughput for Conv2D rather than
only reducing reported DDR bytes.

### Phase 7: Full Runtime Migration

- Extend ONNX lowering and quantization metadata for the supported INT4 Q/DQ
  profile.
- Update model packaging, PS validation, descriptors, buffer sizing, and
  diagnostics.
- Convert bias/requantization/activation/pooling tensor paths to packed INT4 or
  explicitly document INT8 boundaries where conversion is retained.
- Run full virtual CNN comparison, 100 repeated board inferences, accuracy
  evaluation, timing closure, and resource/power reports.

Exit gate: accuracy is within the agreed INT4 quantization tolerance, all board
runs are stable, 100 MHz timing closes, and measured end-to-end latency improves
over Checkpoint-1.

## 6. Decision Matrix

| Question | Default hypothesis | Required evidence |
|---|---|---|
| LUT or DSP INT4 PE? | LUT-only is the simplest first candidate | OOC synthesis and power for full 8x8 |
| Can one DSP compute multiple products? | Only with explicit packed/shared-operand arithmetic | Exhaustive signed proof and post-synth test |
| Is one HP port enough? | No for continuous peak feed at 100 MHz | Board payload-bandwidth counter |
| Should buffers disappear? | Full tile buffers become optional; small FIFOs remain | Stall and reuse measurements |
| What should arrays share? | Share A across adjacent N tiles for Conv2D | Layer-shape cost model and byte counters |
| Should FP4 be included now? | No | Dense INT4 path must first be stable and measured |
| Should sparse support be included? | No | Revisit only with model sparsity and bandwidth evidence |

## 7. Temporary Success Targets

- Functional: bit-exact simulation for all signed INT4 products and golden
  Conv2D/GEMM cases.
- Timing: non-negative WNS at 100 MHz after full implementation.
- Transport: at least 70% payload utilization of each enabled AXI port on long,
  aligned transfers.
- Sharing: approximately 25% lower operand bytes in shared-A/shared-B mode for
  balanced full tiles.
- Compute: initiation interval of one K step per cycle once the array is filled.
- Diagnostics: every optimization report includes latency, effective bandwidth,
  array utilization, starvation cycles, and resource utilization.
- Deployment: 100 consecutive inferences without protocol, DMA, or numerical
  failures.

These are stage gates, not assumed outcomes. The final architecture is selected
from board measurements, with end-to-end CNN latency taking priority over peak
GOPS.

## 8. Initial Compute-Core Measurement

The first LUT-only implementation was synthesized out of context for the
XC7Z020 at a 10 ns clock constraint. Both measurements use the same Vivado
2024.2 flow:

| Compute core | LUT | FF | DSP | Synthesized WNS |
|---|---:|---:|---:|---:|
| Baseline single 16x16 INT8 mixed array | 11,572 | 13,210 | 220 | +4.293 ns |
| New dual 8x8 INT4 LUT-only array | 7,587 | 7,296 | 0 | +5.917 ns |

The initial dual INT4 core reduces LUT use by 34.4%, FF use by 44.8%, releases
all 220 DSP blocks, and improves synthesized timing margin by 1.624 ns. This is
not a precision-only comparison: the INT4 design has 128 PEs while the INT8
baseline has 256 PEs, so peak MAC throughput is halved at the same clock.

The result establishes that the dual INT4 core fits comfortably as an isolated
block. It does not yet establish an end-to-end speedup. Placement, the DDR
stream engine, array utilization, and model accuracy remain required gates.

The initial virtual validation also passes:

- exhaustive signed INT4 PE multiplication: all 256 operand pairs;
- one LUT-only 8x8 systolic array: 64 golden INT32 outputs;
- dual 8x8 arrays in independent, shared-A, and shared-B modes.

The focused regression is reproducible with
`hardware/vivado/run_int4_compute_tests.tcl`. The broad legacy runner was also
updated to include these benches, although its compile-every-source invocation
can spend an excessive amount of time in `xvlog`; this behavior predates the
focused INT4 source set.

### Packed Tile-I/O Migration

The next memory-to-compute layer is implemented with a row-aligned packed INT4
ABI:

- even columns use the low nibble and odd columns use the high nibble;
- one packed loader read produces two signed cache operands;
- two requantized tile values coalesce into one writer byte;
- odd-width rows zero-fill the unused high nibble;
- the dual 8x8 tile core supports independent, shared-A, and shared-B runs.

Focused XSim passes six tests: exhaustive PE, one array, dual arrays, packed
loader, packed odd-tail writer, and the tile-backed dual GEMM core. The loader
test converts 64 DDR bytes into 128 operands, and the seven-column writer emits
32 row-aligned bytes for 56 logical values.

The expanded `gemm_dual_8x8_int4_core` OOC result is:

| LUT | Logic LUT | LUTRAM | FF | DSP | Synthesized WNS |
|---:|---:|---:|---:|---:|---:|
| 9,045 | 8,789 | 256 | 7,436 | 0 | +5.797 ns |

The active bitstream remains on the INT8 path until the bursty Conv2D loader,
requantizer, tensor operators, controller, and PS/host ABI are migrated.

## 9. Cross-Stack INT4 Migration Status

The software ABI and one complete PL CNN tile are now implemented.

- `DType.INT4 = 4` is shared by the host protocol and firmware.
- Tensor storage uses `ceil(elements / 2)` bytes; byte offsets remain byte
  offsets in every network and firmware descriptor.
- The host packs and unpacks signed nibbles, quantizes/dequantizes INT4, and
  preserves logical tensor shapes on reads.
- `import_onnx_model(..., target_dtype="int4")` accepts signed Q/DQ QAT models
  whose INT8 containers are already constrained to `-8..7`. It rejects wider
  values rather than silently clipping them and invalidating Q/DQ scales.
- Firmware validates packed row strides, selects 8x8 tiles and two arrays, and
  implements CPU-reference INT4 Conv2D, requantization, and activation paths.
- `int4_cnn_tile_pipeline.sv` composes dual 8x8 GEMM, per-channel INT32 bias,
  pipelined truncating requantization, saturation, and ReLU/LeakyReLU.

Current regression results:

| Layer | Result |
|---|---:|
| Host unit and virtual model tests | 53 passed, 2 optional-dependency skips |
| Firmware protocol/operator tests | 53 passed |
| Focused XSim INT4 tests | 9 passed |
| CNN tile pipeline output comparison | 128/128 outputs passed |

The verified host-side smoke graph is:

```python
package = budgetcuda.build_tiny_int4_cnn_package(
    (1, 1, 3, 3),
    weights,
    bias_values=bias,
    multiplier=multiplier,
    right_shift=right_shift,
)
model = device.load_model(package)
output = model.run(int4_input)
```

For a PyTorch QAT ONNX model:

```python
package = budgetcuda.import_onnx_model(
    "model_qdq.onnx",
    target_dtype="int4",
)
model = device.load_model(package)
```

### Board Deployment Boundary

The active board-facing wrapper now instantiates
`bcuda_dual_int4_ddr_top`. The final Vivado manifest contains only the packed
INT4 compute and CNN execution path. It supports:

- arbitrary packed INT4 GEMM through two 8x8 LUT systolic arrays;
- packed Conv2D im2col loading with INT32 bias preload;
- pipelined INT32-to-packed-INT4 requantization;
- eight-lane packed ReLU and shift-only LeakyReLU;
- packed signed-INT4 MaxPool and Average/GlobalAveragePool;
- packed nearest-neighbor upsample; and
- packed NCHW channel concat with up to three inputs.

The full-top XSim test applies aperiodic DDR command, response, and write
backpressure. It passes irregular GEMM, padded biased Conv2D, odd-length
requantization, odd-length ReLU/LeakyReLU, MaxPool, GlobalAveragePool,
upsample, and concat golden comparisons.

The complete DDR top passes out-of-context synthesis for the PYNQ-Z2
`xc7z020clg400-1` target at 100 MHz:

| LUTs | FFs | RAMB18 | DSP | 100 MHz WNS |
|---:|---:|---:|---:|---:|
| 13,147 | 12,048 | 4 | 30 | +0.396 ns |

Both systolic arrays remain LUT-only. The 30 DSP48E1 slices are used by
pipelined address generation, result address calculation, and requantization,
leaving 190 DSP48E1 slices for later experiments. The current OOC worst path
remains an existing engine dimension multiply path rather than a migrated CNN
operator.

The remaining board deployment sequence is:

1. Regenerate the PS/PL block design and run full implementation timing.
2. Generate and deploy a new bitstream.
3. Run layer-by-layer board comparison against the host INT4 golden model.
4. Run the 100-inference accuracy and stability test.
5. Measure per-operator DDR efficiency and replace single-byte tensor reads
   with burst/cache-line fetches where diagnostics show starvation.

Checkpoint-1 remains the immutable INT8 fallback. Legacy INT8 RTL stays in the
development regression until its benches and documentation are moved together
to `FPGA/deprecated`; it is not part of the final INT4 bitstream manifest.
