# BudgetCUDA Checkpoint 1

Presentation checkpoint for July 28, 2026.

## Project

BudgetCUDA is an end-to-end micro-accelerator stack for the PYNQ-Z2:

1. The Python host library imports and lowers quantized ONNX models.
2. The host sends models, inputs, and graph commands over Ethernet.
3. PS firmware manages model memory, dispatch, diagnostics, and PL commands.
4. The PL executes DDR-backed tiled GEMM and CNN tensor operations.
5. Results and per-layer diagnostics return through the host API.

The intended user experience is a small general-purpose inference device, not
a hardwired CNN: select the PYNQ device, load an ONNX model, and run inference.

## Demonstrated Checkpoint

- Target: PYNQ-Z2, XC7Z020.
- Clock: 100 MHz.
- Post-route WNS: +0.089 ns.
- Compute: one mixed 16x16 INT8 systolic array.
- PE allocation: 180 DSP MACs and 76 LUT MACs.
- Theoretical peak: 25.6 GMAC/s or 51.2 GOPS when multiply and add are counted
  separately.
- Resource use: 23,875 LUTs (44.88%), 21,607 registers (20.31%), 5 BRAM tiles
  (3.57%), and 220 DSPs (100%).
- Full active RTL regression: 17/17 simulations pass.
- FPGA manager state on the board: operating.
- BudgetCUDA PS service state: active.

## Board CNN Result

- Fruit classifier output logits: `[-25, 33, -20]`.
- Firmware layer time: 224.595 ms.
- Host inference time: 260.468 ms.
- Conv2D utilization: 0.106%, 0.149%, and 0.213%.
- Firmware arena high-water use: 565,568 bytes.

The full diagnostic exports are under
`FPGA/tests/hardware/diagnostics/tileloader_control_pipeline`.

## Memory-Wall Calculation

For one 16x16x16 tile:

- A tile: 256 bytes.
- B tile: 256 bytes.
- Total input: 512 bytes.
- Compute duration: approximately 49 cycles.
- Current byte-wide load floor: 512 cycles.
- Compute duty-cycle ceiling before other overhead: 49 / 561 = 8.7%.
- Useful-operation utilization ceiling: 8,192 / (561 * 512) = 2.85%.
- Refill requirement without reuse: 1.045 GB/s.
- Refill requirement with persistent B: 522 MB/s.
- Refill requirement with fourfold A and B reuse: 261 MB/s.

The local BRAM wave memories can supply 32 INT8 operands per cycle, or
3.2 GB/s at 100 MHz. The present shared path is limited by a 32-bit,
single-outstanding AXI bridge and an 8-bit loader/core interface.

## Post-Checkpoint Direction

Feedback from Etched led to the following priorities:

1. Build a buffer-minimal DDR streaming path for low-reuse operations.
2. Keep optional BRAM residency and reuse for CNN/YOLO and other high-reuse
   workloads.
3. Replace the mixed 16x16 array with two independently schedulable 8x8
   DSP-only arrays.
4. Move the arithmetic and model contract from INT8 toward INT4.
5. Treat sparse execution as optional future work, not a core dependency.
6. Present the engineering process, measurements, failed experiments, and
   verification evidence alongside final results.

## Final Board Artifacts

- Bitstream: `FPGA/hardware/bcuda_pynq_z2.bit`
- Hardware handoff metadata: `FPGA/hardware/bcuda_pynq_z2.hwh`
- FPGA-manager image SHA-256:
  `D91850F64F225F488F234606F8F3FA494B69E7D00C621134E7F56792D868AC6C`

This branch is the stable presentation checkpoint. Subsequent architectural
work should begin from `main`.
