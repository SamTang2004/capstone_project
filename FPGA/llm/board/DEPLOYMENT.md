# Sparse engine: physical board deployment

This research overlay follows the connection, matching-artifact, service ownership,
and recovery rules in [../../RUN_GUIDE.md](../../RUN_GUIDE.md). It is separate from
both the DeiT release and the older LLM GEMM/DDR overlay.

## Scope

- Target: PYNQ-Z2, xc7z020clg400-1, AXI-Lite base 0x43C90000.
- Compute and operand memories target 200 MHz on FCLK0. AXI3 protocol conversion
  runs at 100 MHz on FCLK1, followed by Vivado's asynchronous AXI-Lite clock
  converter. Both clocks reserve 0.300 ns setup uncertainty.
- Uses a separate pipelined `llm_sparse_board_control.sv`; shared production
  `hardware/rtl/axi_lite_control.sv` remains unchanged.
- Historical full-design attempts failed setup at -1.947, -0.878 and -0.359 ns.
  None was programmed. The earlier 100 MHz release remains available as a baseline.
- The final complete 200 MHz implementation passes setup +0.004 ns / hold +0.024 ns,
  CDC and bus-skew checks. Physical validation passes 96 commands at 200 MHz and
  32 at 100 MHz with the same image and identical counters. See
  [the result record](../research/SPARSE_BOARD_200_RESULTS.md).
- CPU uploads 4 KiB activation storage and 256 KiB weight storage into BRAM.
- Eight weight banks feed the actual sparse activation-sharing engine. Activation
  frames are assembled from synchronous 64-bit BRAM reads, one outstanding frame.
- Weight requests and BRAM outputs are registered. A 16-entry response queue
  reserves space for at most 12 outstanding requests, avoiding a combinational
  ready chain through memory enables. Activation addresses and outputs are also
  registered; masks and end markers follow the same latency.
- The actual metadata producer runs before each consumer invocation. Preparation
  and consumer cycles are reported separately. No input upload is in either count.
- These are local-memory hardware measurements, not DDR bandwidth or LLM inference
  results. They need not match the earlier eight-outstanding-request simulation.
- Separate BIT/HWH/XSA and reports are under `build/sparse_release200`. The build
  refuses bitstream export on negative setup or hold slack. No timing fixture or
  DUT false paths are used in this complete PS/PL build.

## Build and validate locally

```powershell
& FPGA/llm/sim/run_sparse_board.ps1
& FPGA/llm/sim/run_sparse_board_control.ps1
& 'C:/Xilinx/Vivado/2024.2/bin/vivado.bat' -mode batch -source FPGA/llm/vivado/build_sparse_board.tcl -log FPGA/llm/build/sparse_board200_build.log -journal FPGA/llm/build/sparse_board200_build.jou -tclargs 200
python FPGA/llm/board/release_manifest.py FPGA/llm/build/sparse_release200
```

The board-memory simulation covers 25 commands with exact signed INT32 results,
all four skip/reuse modes, maximum K/row count, partial blocks/tiles, empty inputs,
thresholds, cancellation at four weight-pipeline positions, sustained response
backpressure, and invalid-shape rejection. The AXI test checks 33 writes and 51
reads, independent AW/W arrivals, strobes, stalled responses and partial-write reset.
Vivado project builds need
Windows CScript/child-process access. The vendored PS preset retains its documented
negative DDR-DQS delay warnings; they must not be mistaken for a clean board-wide
DDR qualification.

## Deploy

1. SSH to `xilinx@192.168.2.99`. Authenticate interactively; no password is stored
   in scripts. Inspect FPGA state and service status.
2. Keep the old BIT/HWH and services intact. Stop any active accelerator daemon
   before programming or direct MMIO access. Do not enable the old services against
   this new research ABI.
3. Create `/home/xilinx/sparse_research_200` and copy the matched
   `sparse_pynq_z2.bit`, `sparse_pynq_z2.hwh`, release manifest and
   `validate_sparse_board.py` there. Record SHA256 hashes and source provenance.
4. Run as root (PYNQ/XRT device access requires it):

```sh
cd /home/xilinx/sparse_research_200
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 validate_sparse_board.py --bit sparse_pynq_z2.bit --clock-mhz 100 --repeat 1 --output board_results_pipeline100.json
sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 validate_sparse_board.py --bit sparse_pynq_z2.bit --clock-mhz 200 --repeat 3 --output board_results_200.json
```

The runner verifies manifest hashes and rejects clocks above the release timing
qualification before programming. It programs the overlay, sets FCLK0 using PYNQ, checks the actual clock and
ABI, uploads deterministic signed operands, and checks every row against NumPy
INT64 goldens reduced modulo 2^32. It checks metadata and activation request counts.
Results save after every case; a timeout cancels the command. Preserve stdout and
JSON, then retrieve both into `research/` with the exact release manifest.
The two-clock image keeps FCLK1 at 100 MHz for both runs. Compare identical
hardware counters between 100 and 200 MHz before interpreting elapsed-time gains.
Run `python FPGA/llm/board/report_board_pipeline.py` after placing the new JSON
files in `research/board200/`; it verifies matching outputs and counters against
the original `research/board100/` data before generating the comparison.

## Register ABI 0x53505202

All registers are 32-bit. Control at 0x00 uses toggle bits (change the bit once per
action): bit 0 launch; bit 1 activation-word upload; bit 2 weight-word upload;
bit 3 cancel/reset the engine and adapters. Uploads are only legal while idle.

| Offset | Meaning |
| --- | --- |
| 0x04 | Status: bit 0 busy, bit 1 done, bit 2 error |
| 0x08 | K, 0..4096 |
| 0x0c | Rows 1..64 in bits 6:0; shift 11:8; skip bit 16; reuse bit 17 |
| 0x10 | Upload word address (A 0..1023; W row*1024 + column/4) |
| 0x14 | Four signed byte operands, little-endian |
| 0x18 | Result row selector 0..63 |
| 0x40 | Identity 0x53505202 |
| 0x44 | Consumer execution cycles |
| 0x48 | Metadata preparation cycles, including wrapper startup/drain |
| 0x4c | Activation frame requests |
| 0x50 | Weight requests, including zero-mask end markers |
| 0x54 | Selected INT32 result (registered selector; allow an AXI round trip) |
| 0x58 / 0x5c | Generated 64-bit block map |
| 0x60 | Consumer scan-state cycles |

Weight bank is row modulo 8; tile is row divided by 8. CPU word uploads preserve
row-major order; the hardware reads eight rows at the same activation index.
Counter widths are 32-bit. The runner verifies completion and reads results before
another command can overwrite them. Cancellation invalidates in-flight pipeline
state and leaves operand memories intact. Registered cancellation takes effect
one cycle after detection; result selection is followed by an AXI round trip.

## Recovery

Do not use the old TCP daemon against this ABI. To restore the previous DeiT image,
use the retained matched `/opt/budgetcuda/overlay/bcuda_pynq_z2.bit/.hwh` through the
existing FPGA setup service, then start the daemon only if the user wants the old
service restored. Do not change boot enablement as part of this temporary test.
The earlier sparse 100 MHz image, manifest and results are also retained under
`/home/xilinx/sparse_research_200/baseline100/` and locally in `build/sparse_release100`.
