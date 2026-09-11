# BudgetCUDA DeiT-Tiny run guide

This is the operational guide for the `DeiT_Tiny` branch and the timing-closed
PYNQ-Z2 release. It covers the shortest path to inference, full deployment of a
matched bitstream and daemon, benchmark reporting, and recovery after a board
or service failure.

For a clean host and board installation with no existing BudgetCUDA deployment,
start with `FPGA/DEPLOYMENT_FROM_GROUND_ZERO.md`.

The validated implementation is the W4A8 DeiT-Tiny accelerator with optional
balanced HeatViT token pruning. At 100 MHz, the reference run averages
416.63 ms for the resident device graph and 440.27 ms host-to-host.

## 1. Validated release contract

Keep these components from the same branch revision:

- Python package under `FPGA/budgetcuda/`;
- firmware headers and daemon under `FPGA/firmware/`;
- `FPGA/hardware/bcuda_pynq_z2.bit`;
- `FPGA/hardware/bcuda_pynq_z2.hwh`; and
- the W4A8/HeatViT manifest with all binary files it references.

The bitstream and daemon share command flags, register fields, layouts, and
diagnostic-record formats. Mixing versions can hang the graph or produce
plausible but incorrect output. The validated implementation commit is
`cd7ef8f`; later documentation-only commits do not change the release binary.

The board services configure:

- PYNQ-Z2 (`xc7z020clg400-1`);
- FCLK0 at 100 MHz;
- native 64-bit HP0 access;
- a 64 MiB accelerator arena; and
- the BudgetCUDA TCP service at `192.168.2.99:9000` by default.

## 2. Quick run on an already-installed board

### 2.1 Start the board and check readiness

After applying power, allow Linux and the FPGA service to finish. Programming
the overlay is ordered after PYNQ's `bootpy.service`, so the board can take
longer to become ready than a basic Linux login.

From Windows PowerShell:

```powershell
Test-Connection 192.168.2.99 -Count 2
Test-NetConnection 192.168.2.99 -Port 9000
```

Then inspect the board:

```powershell
ssh xilinx@192.168.2.99 `
  "cat /sys/class/fpga_manager/fpga0/state; systemctl is-active budgetcuda-fpga.service; systemctl is-active budgetcuda.service; ss -ltn | grep ':9000'"
```

Expected output contains:

```text
operating
active
active
LISTEN ... :9000
```

If SSH works but port 9000 is not listening, follow the service recovery steps
in Section 9 instead of immediately power cycling.

### 2.2 Prepare the host environment

From the repository root:

```powershell
& .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -r testing\deit_tiny\requirements.txt
```

PyTorch and torchvision must be installed separately with builds appropriate
for the host's CPU or CUDA environment. They are needed for training and model
export; an already-exported manifest can run with the lightweight host stack.

Confirm that the ignored model and image artifacts are present:

```powershell
Test-Path testing\deit_tiny\artifacts\heatvit_w4a8_balanced\manifest.json
Test-Path 'testing\deit_tiny\data\cache_256\FreshApple\FreshApple (1).jpg'
```

Both must print `True`. Dataset images, checkpoints, and exported model
artifacts are intentionally excluded from Git and must be restored or generated
on a fresh checkout.

### 2.3 Run the reference production benchmark

```powershell
& .\.venv\Scripts\python.exe testing\deit_tiny\run_budgetcuda_deit.py `
  'testing\deit_tiny\data\cache_256\FreshApple\FreshApple (1).jpg' `
  --manifest testing\deit_tiny\artifacts\heatvit_w4a8_balanced\manifest.json `
  --address 192.168.2.99 `
  --port 9000 `
  --timeout 120 `
  --clock-hz 100000000 `
  --heatvit `
  --repeat 2 `
  --production-repeat 3 `
  --compact `
  --cycle-report testing\deit_tiny\results\fpga_cycle_report_current.json
```

Do not interrupt the initial upload merely because it is slower than one
inference. `runner.load()` allocates the complete graph and uploads the model
once; every repeat then uses the resident CUDA-style execution path.

The reference image should produce:

```text
class_name: apple_fresh
logits: [1, -3, 1, 1, 0, -6]
```

The validated performance anchors are:

- production device-graph mean: 416.63 ms;
- production host-to-host mean: 440.27 ms;
- exact diagnostic window: 41,686,928 cycles, or 416.869 ms; and
- effective diagnostic throughput: 4.335 GOPS.

Diagnostics add counter initialization and readback, so use
`production_repeat_seconds` and `production_stage_seconds` for steady-state
application latency. Use the saved cycle report for module attribution.

To compare isolated latency against sustained resident-model throughput, run:

```powershell
& .\.venv\Scripts\python.exe testing\deit_tiny\run_continuous_benchmark.py `
  'testing\deit_tiny\data\cache_256\FreshApple\FreshApple (1).jpg' `
  --manifest testing\deit_tiny\artifacts\heatvit_w4a8_balanced\manifest.json `
  --address 192.168.2.99 `
  --clock-hz 100000000 `
  --warmup 2 `
  --iterations 20 `
  --heatvit `
  --output testing\deit_tiny\results\fpga_continuous_benchmark.json
```

The current protocol serializes complete graphs. This benchmark continuously
submits resident-model executions and measures sustained images/second, but it
does not claim cross-image FPGA overlap. Its wall-clock occupancy estimate uses
the exact busy-cycle count from the isolated diagnostic graph and the elapsed
time of the diagnostics-disabled production stream.

### 2.4 Select the pruning mode

The same HeatViT-capable manifest supports both modes:

```powershell
# Balanced token pruning: 197 -> 168 -> 133 -> 80
--heatvit

# Full token sequence; selectors remain loaded but are not scheduled
--no-heatvit
```

The sub-500 ms reference requires `--heatvit`. Always record the selected mode
with performance or accuracy results; pruned and unpruned timings are not
interchangeable.

## 3. Restore or regenerate model artifacts

The quickest reproducible path is to restore the archived directory
`testing/deit_tiny/artifacts/heatvit_w4a8_balanced/` intact. The manifest uses
relative filenames, so copying only `manifest.json` is insufficient.

To regenerate it, first prepare the licensed Fresh and Rotten Fruits dataset as
described in `testing/deit_tiny/README.md`. Place the publisher's original
folders below:

```text
testing/deit_tiny/data/original/Original Image/
```

Then create the deterministic cache and split audit:

```powershell
& .\.venv\Scripts\python.exe testing\deit_tiny\prepare_dataset.py
```

After producing the per-tensor W4A8 checkpoint, export the base accelerator
model:

```powershell
& .\.venv\Scripts\python.exe testing\deit_tiny\export_budgetcuda.py `
  --checkpoint testing\deit_tiny\artifacts\w4a8_per_tensor\best_fake_quant.pt `
  --output-dir testing\deit_tiny\artifacts\budgetcuda_w4a8 `
  --weight-bits 4
```

Train and export the balanced W8-selector profile:

```powershell
& .\.venv\Scripts\python.exe testing\deit_tiny\train_heatvit_pruner.py `
  --checkpoint testing\deit_tiny\artifacts\w4a8_per_tensor\best_fake_quant.pt `
  --base-export testing\deit_tiny\artifacts\budgetcuda_w4a8 `
  --output-dir testing\deit_tiny\artifacts\heatvit_w4a8_balanced `
  --selector-blocks 2 5 8 `
  --output-tokens 168 133 80 `
  --selector-bits 8 `
  --device cuda
```

Before deployment, preserve the complete output directory and its accuracy,
training, conformance, and manifest records. Retraining may produce different
weights and therefore should not be expected to reproduce the reference logits
without matching the saved checkpoint and seed.

## 4. Verify source before rebuilding hardware

Run host and firmware tests independently. Combining them in one pytest command
can cause duplicate test-module collection on this repository layout.

```powershell
& .\.venv\Scripts\python.exe -m pytest FPGA\tests\host -q
& .\.venv\Scripts\python.exe -m pytest FPGA\tests\firmware -q
```

Validated totals are 90 host tests passed with one optional skip and 65
firmware tests passed.

Run the complete RTL regression:

```powershell
& 'C:\Xilinx\Vivado\2024.2\bin\vivado.bat' -mode batch `
  -source FPGA\hardware\vivado\run_hardware_tests.tcl
```

All 27 benches must report `golden trace passed`, including the shared
LayerNorm/Softmax equivalence test and integrated Transformer DDR top.

## 5. Build the timing-qualified PYNQ-Z2 image

Use Vivado 2024.2 from the repository root:

```powershell
& 'C:\Xilinx\Vivado\2024.2\bin\vivado.bat' -mode batch `
  -source FPGA\hardware\vivado\build_pynq_z2_bitstream.tcl
```

The deployable files are:

```text
FPGA/hardware/bcuda_pynq_z2.bit
FPGA/hardware/bcuda_pynq_z2.hwh
```

Review the newly generated reports rather than assuming the previous build's
slack:

```powershell
Select-String FPGA\hardware\pynq_z2_timing_impl.rpt -Pattern 'WNS\(ns\)'
Select-String FPGA\hardware\pynq_z2_utilization_impl.rpt `
  -Pattern 'Slice LUTs|Slice Registers|Block RAM Tile|DSPs'
```

Release requirements are non-negative WNS and WHS, zero failed or unrouted
nets, and no implementation DRC errors. The validated image closes at WNS
`+0.006 ns` and WHS `+0.033 ns` with 50.18% LUT, 28.89% registers, 30% BRAM,
and 75% DSP usage.

## 6. Deploy a matched bitstream and daemon

The current validated deployment uses the BIT/HWH overlay pair. Do not use an
older CNN service, register map, or FPGA-manager payload with this branch.

The commands below use Windows OpenSSH. Replace the board address if needed.

### 6.1 Create temporary staging directories

```powershell
ssh xilinx@192.168.2.99 `
  "rm -rf -- /tmp/bcuda-src /tmp/bcuda-deploy; mkdir -p /tmp/bcuda-src /tmp/bcuda-deploy"
scp -r FPGA/firmware/include FPGA/firmware/src FPGA/firmware/linux `
  xilinx@192.168.2.99:/tmp/bcuda-src/
```

Only these two explicitly named temporary staging directories are removed. The
installed release and its backup under `/opt/budgetcuda` are untouched.

### 6.2 Build the ARM daemon natively on the board

```powershell
ssh xilinx@192.168.2.99 `
  "sh /tmp/bcuda-src/linux/build_server.sh /tmp/bcuda-deploy/budgetcuda-server"
```

### 6.3 Copy the release overlay and service payload

```powershell
scp FPGA/hardware/bcuda_pynq_z2.bit `
  FPGA/hardware/bcuda_pynq_z2.hwh `
  FPGA/firmware/linux/budgetcuda-fpga-setup.sh `
  FPGA/firmware/linux/budgetcuda_overlay.py `
  FPGA/firmware/linux/pynq_fclk0.py `
  FPGA/firmware/linux/pynq_afi_hp0.py `
  FPGA/firmware/linux/budgetcuda-fpga.service `
  FPGA/firmware/linux/budgetcuda.service `
  FPGA/firmware/linux/deploy_staged_board.sh `
  xilinx@192.168.2.99:/tmp/bcuda-deploy/
```

### 6.4 Install and start the release

```powershell
ssh -t xilinx@192.168.2.99 `
  "sudo sh /tmp/bcuda-deploy/deploy_staged_board.sh /tmp/bcuda-deploy"
```

The deployment script stops both services, backs up the preceding installation
under `/opt/budgetcuda/backup-pre-heatvit`, installs the matched files, programs
the FPGA, configures FCLK0 and HP0, then starts the daemon. Success requires
both systemd units to be active.

Record hashes for every benchmarked release:

```powershell
Get-FileHash FPGA\hardware\bcuda_pynq_z2.bit -Algorithm SHA256
Get-FileHash FPGA\hardware\bcuda_pynq_z2.hwh -Algorithm SHA256
git rev-parse HEAD
```

## 7. Board smoke tests after deployment

First verify the service-based W8A8 and W4A8 Transformer graphs:

```powershell
& .\.venv\Scripts\python.exe FPGA\tests\hardware\pynq_transformer_graph_smoke.py `
  --address 192.168.2.99 --mode both --timeout 30
```

Expected final lines include:

```text
PASS: W8A8 MLP and complete three-head attention graphs
PASS: W4A8 MLP and complete three-head attention graphs
```

Then verify both selector weight modes:

```powershell
& .\.venv\Scripts\python.exe FPGA\tests\hardware\pynq_heatvit_selector_smoke.py `
  --address 192.168.2.99 --mode both --timeout 30
```

Only after these smokes pass should the full resident model be loaded. Direct
MMIO or allocator tests must not run concurrently with the system daemon; stop
the service first if a low-level board diagnostic is required.

## 8. Mandatory benchmark record

Every FPGA performance run must save and report:

- branch and commit;
- bitstream and HWH hashes;
- manifest path and pruning mode;
- board clock;
- input image;
- class and raw logits;
- production host and device-graph repeats;
- total, busy, and memory-stall cycles;
- every module's cycle count and graph share;
- GEMM Load A, Load B, compute, and write counters; and
- timing and utilization reports for the deployed image.

The reference final module distribution is:

| Module | Cycles | Time | Graph share |
|---|---:|---:|---:|
| GEMM | 25,931,472 | 259.31 ms | 62.21% |
| Softmax | 4,207,461 | 42.07 ms | 10.09% |
| Residual add | 3,632,378 | 36.32 ms | 8.71% |
| LayerNorm | 3,297,462 | 32.97 ms | 7.91% |
| Token concatenation | 2,947,054 | 29.47 ms | 7.07% |
| Token embedding | 1,098,517 | 10.99 ms | 2.64% |
| Token pruning | 572,584 | 5.73 ms | 1.37% |

The canonical raw reference is
`testing/deit_tiny/results/fpga_cycle_report_100mhz_fused_shared_correct.json`.

## 9. Recovery and troubleshooting

### Board responds to SSH but not port 9000

```powershell
ssh xilinx@192.168.2.99 `
  "systemctl --no-pager --full status budgetcuda.service; journalctl -u budgetcuda.service -n 100 --no-pager"
```

Restart only the daemon first:

```powershell
ssh -t xilinx@192.168.2.99 "sudo systemctl restart budgetcuda.service"
```

This clears all resident allocations, so the host must reload the model.

### FPGA service failed or state is not `operating`

```powershell
ssh xilinx@192.168.2.99 `
  "systemctl --no-pager --full status budgetcuda-fpga.service; journalctl -u budgetcuda-fpga.service -n 100 --no-pager; cat /sys/class/fpga_manager/fpga0/state"
```

Reprogram the overlay and then restart the daemon:

```powershell
ssh -t xilinx@192.168.2.99 `
  "sudo systemctl stop budgetcuda.service; sudo systemctl restart budgetcuda-fpga.service; sudo systemctl start budgetcuda.service"
```

### Boot takes a long time

Do not repeatedly remove power while the SD card may still be active. The FPGA
unit waits for PYNQ boot initialization before loading the final overlay. Check
`bootpy.service`, `budgetcuda-fpga.service`, and their journals separately. If
networking is not yet available, use the FTDI UART helper at 115200 baud:

```powershell
& .\.venv\Scripts\python.exe testing\deit_tiny\scan_ftdi_uart.py `
  --serial 1234-tulB --seconds 10 --prompt
```

### Board freezes or disconnects during testing

- Use one control path at a time. Do not run direct `/dev/mem`, XRT, or XLNK
  tests while the TCP daemon owns the accelerator.
- Stop the daemon before reprogramming the FPGA.
- Confirm stable board power and Ethernet before attributing a disconnect to
  RTL.
- Inspect `journalctl`, `dmesg`, FPGA state, FCLK0, and HP0 configuration after
  reconnecting.
- A failed or interrupted graph may leave the PL busy. Restart the FPGA service
  followed by the daemon before retrying.
- Repeated allocation failures usually mean stale resident handles or CMA
  pressure. Restarting the daemon clears its 64 MiB arena.

### Correct class but unexpected logits or latency

- Confirm `--heatvit` versus `--no-heatvit`.
- Confirm the full artifact directory matches its manifest hashes.
- Confirm host, daemon, BIT, and HWH all come from the same branch revision.
- Confirm FCLK0 is 100 MHz; cycle counts can match while elapsed time differs
  if the clock is wrong.
- Compare the per-module report rather than relying only on host wall time.

### Clean shutdown

The runner frees its remote buffers when it exits normally. To power down the
board safely:

```powershell
ssh -t xilinx@192.168.2.99 "sudo shutdown -h now"
```

Wait for shutdown to complete before removing power.

## 10. Release checklist

- [ ] Host and firmware tests pass.
- [ ] All 27 RTL golden traces pass.
- [ ] Fresh implementation has non-negative setup and hold slack.
- [ ] Daemon, BIT, HWH, and host protocol come from one revision.
- [ ] FPGA state is `operating`; both services are active; port 9000 listens.
- [ ] W8A8/W4A8 graph and HeatViT selector smokes pass.
- [ ] Full reference inference preserves class and exact logits.
- [ ] Production repeats and a complete module/GEMM cycle report are saved.
- [ ] Commit, clock, manifest, and release hashes accompany the result.

The design history and controlled optimization measurements are documented in
`testing/deit_tiny/results/FPGA_IMPLEMENTATION_TIMELINE.md`.
