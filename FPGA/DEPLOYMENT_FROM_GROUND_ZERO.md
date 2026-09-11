# BudgetCUDA DeiT-Tiny deployment from ground zero

This guide rebuilds the complete PYNQ-Z2 deployment from a clean Windows host
and a clean, bootable PYNQ Linux image. It ends with a matched 100 MHz FPGA
overlay, board daemon, W4A8 DeiT-Tiny/HeatViT model, smoke tests, exact cycle
diagnostics, and a sustained 20-image benchmark.

Use `FPGA/RUN_GUIDE.md` for normal operation after this procedure succeeds.

## 1. Release contract and required material

The host package, ARM daemon, BIT/HWH pair, and model manifest must belong to
the same `DeiT_Tiny` revision. Do not combine this release with an older CNN
daemon, bitstream, register map, or Python package.

Required hardware and connections:

- PYNQ-Z2 (`xc7z020clg400-1`) with a stable power supply;
- a bootable SD card containing a PYNQ Linux image;
- Ethernet between the host and board; and
- the board's FT2232H USB/JTAG/UART connection for recovery.

Required host software:

- Git and Windows OpenSSH (`ssh` and `scp`);
- 64-bit Python 3.12 and PowerShell;
- AMD/Xilinx Vivado 2024.2 for rebuilding or JTAG inspection; and
- the 64-bit FTDI D2XX driver (`C:\Windows\System32\ftd2xx.dll`) for direct
  UART recovery when Windows does not create a COM port.

The checked-in release pair at the time this guide was written has these
SHA-256 hashes:

```text
bcuda_pynq_z2.bit  5863B5C1F7C87AC9E1C88087EDA47C73BAB5E426E2918E0DE06427AAFB92D0DA
bcuda_pynq_z2.hwh  2FBB3EF754E29A510783846339E80B9D60415E62E23B3AD65A64CC966E2527EC
```

Rebuilding the image creates a new release candidate. Record its hashes and
timing reports rather than expecting the hashes above to remain unchanged.

## 2. Prepare a clean host checkout

In PowerShell, choose a parent directory and clone the repository:

```powershell
git clone https://github.com/SamTang2004/capstone_project.git
Set-Location capstone_project
git fetch origin
git switch DeiT_Tiny
git status --short
git rev-parse HEAD
```

`git status --short` should be empty before building a release. Never discard a
dirty worktree automatically; preserve or commit intentional work first.

Create the Python environment:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r testing\deit_tiny\requirements.txt
```

Training and export additionally require compatible PyTorch and torchvision
builds. They are not required to run an already-exported accelerator manifest.

Run the software tests:

```powershell
& .\.venv\Scripts\python.exe -m pytest FPGA\tests\host -q
& .\.venv\Scripts\python.exe -m pytest FPGA\tests\firmware -q
```

The validated baseline is 90 host tests with one optional skip and 65 firmware
tests. Investigate failures before deploying.

## 3. Restore the model and test image

The dataset, checkpoints, and exported binary tensors are intentionally ignored
by Git. Restore the complete balanced HeatViT export directory, not only its
manifest:

```text
testing/deit_tiny/artifacts/heatvit_w4a8_balanced/
```

Confirm the manifest and reference image:

```powershell
Test-Path testing\deit_tiny\artifacts\heatvit_w4a8_balanced\manifest.json
Test-Path 'testing\deit_tiny\data\cache_256\FreshApple\FreshApple (1).jpg'
```

Both commands must return `True`. To regenerate these assets from training
data, follow the `Reproduce` section of `testing/deit_tiny/README.md` and
Section 3 of `FPGA/RUN_GUIDE.md`.

## 4. Establish a clean PYNQ Linux baseline

Write a compatible PYNQ-Z2 Linux image to the SD card using the image
publisher's documented imaging procedure. The repository does not distribute
or modify the base operating-system image.

The board used for the verified deployment has this compatibility fingerprint:

```text
Operating system  PynqLinux 3.0 (Carlisle), Ubuntu 22.04 base
Kernel            6.6.10-xilinx-v2024.1-g3c0eca68c652
PYNQ Python        3.1.1
FPGA allocator     ZOCL loaded; XRT core library available
```

Use this image family for exact reproduction. A newer image may work, but treat
it as a new platform qualification because FPGA-manager, PYNQ Overlay, XRT,
ZOCL, and CMA behavior are part of the deployment contract.

Boot the board once and confirm these board facilities before installing
BudgetCUDA:

```sh
command -v gcc
test -x /usr/local/share/pynq-venv/bin/python3
test -w /sys/class/fpga_manager/fpga0/firmware
test -r /usr/include/xrt/xrt/xrt_bo.h
test -e /usr/lib/libxrt_coreutil.so
ip -4 address
```

The build script can fall back to the legacy allocator when XRT headers and
libraries are absent, but the validated 64 MiB resident-model deployment uses
the board's XRT/ZOCL path. For release reproduction, restore a matching PYNQ
image if those XRT checks fail.

Configure the board and host Ethernet interfaces on the same subnet. This guide
uses:

```text
Host:  192.168.2.1/24
Board: 192.168.2.99/24
```

If the board uses DHCP or another static address, substitute that address in
every command below. Verify connectivity:

```powershell
Test-Connection 192.168.2.99 -Count 2
Test-NetConnection 192.168.2.99 -Port 22
ssh xilinx@192.168.2.99 "uname -a; ip -4 address"
```

Do not proceed until SSH works reliably.

## 5. Select checked-in hardware or rebuild it

For exact redeployment of the checked-in release, use:

```text
FPGA/hardware/bcuda_pynq_z2.bit
FPGA/hardware/bcuda_pynq_z2.hwh
```

Verify their hashes:

```powershell
Get-FileHash FPGA\hardware\bcuda_pynq_z2.bit -Algorithm SHA256
Get-FileHash FPGA\hardware\bcuda_pynq_z2.hwh -Algorithm SHA256
```

To rebuild from RTL, first run all 27 golden benches:

```powershell
& 'C:\Xilinx\Vivado\2024.2\bin\vivado.bat' -mode batch `
  -source FPGA\hardware\vivado\run_hardware_tests.tcl
```

Then implement the PYNQ-Z2 image:

```powershell
& 'C:\Xilinx\Vivado\2024.2\bin\vivado.bat' -mode batch `
  -source FPGA\hardware\vivado\build_pynq_z2_bitstream.tcl
```

Release only an image with non-negative setup and hold slack, no unrouted nets,
and no implementation DRC errors:

```powershell
Select-String FPGA\hardware\pynq_z2_timing_impl.rpt -Pattern 'WNS\(ns\)'
Select-String FPGA\hardware\pynq_z2_utilization_impl.rpt `
  -Pattern 'Slice LUTs|Slice Registers|Block RAM Tile|DSPs'
```

The last validated route closed at 100 MHz with WNS `+0.006 ns`, WHS
`+0.033 ns`, 50.18% LUT, 28.89% registers, 30% BRAM, and 75% DSP utilization.

## 6. Stage the complete board release

Create explicit temporary directories on the board:

```powershell
ssh xilinx@192.168.2.99 `
  "rm -rf -- /tmp/bcuda-src /tmp/bcuda-deploy; mkdir -p /tmp/bcuda-src /tmp/bcuda-deploy"
```

Copy the firmware source and build script:

```powershell
scp -r FPGA/firmware/include FPGA/firmware/src FPGA/firmware/linux `
  xilinx@192.168.2.99:/tmp/bcuda-src/
```

Build the ARM daemon natively on the board:

```powershell
ssh xilinx@192.168.2.99 `
  "sh /tmp/bcuda-src/linux/build_server.sh /tmp/bcuda-deploy/budgetcuda-server"
```

Copy the matched FPGA and service payload:

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

Optionally compare the staged hashes before installation:

```powershell
ssh xilinx@192.168.2.99 `
  "sha256sum /tmp/bcuda-deploy/bcuda_pynq_z2.bit /tmp/bcuda-deploy/bcuda_pynq_z2.hwh /tmp/bcuda-deploy/budgetcuda-server"
```

## 7. Install, enable, and start BudgetCUDA

Install the release using an interactive SSH session so `sudo` can prompt
safely:

```powershell
ssh -t xilinx@192.168.2.99 `
  "sudo sh /tmp/bcuda-deploy/deploy_staged_board.sh /tmp/bcuda-deploy"
```

On a ground-zero image, explicitly enable both units for later boots:

```powershell
ssh -t xilinx@192.168.2.99 `
  "sudo systemctl enable budgetcuda-fpga.service budgetcuda.service"
```

The installation creates:

```text
/opt/budgetcuda/budgetcuda-server
/opt/budgetcuda/overlay/bcuda_pynq_z2.bit
/opt/budgetcuda/overlay/bcuda_pynq_z2.hwh
/usr/local/sbin/budgetcuda-fpga-setup
/usr/local/lib/budgetcuda/*.py
/etc/systemd/system/budgetcuda-fpga.service
/etc/systemd/system/budgetcuda.service
```

The FPGA unit programs the matched overlay, selects native 64-bit HP0, and sets
FCLK0 to 100 MHz. The daemon then reserves its 64 MiB accelerator arena and
listens on TCP port 9000.

Verify the installed state:

```powershell
ssh xilinx@192.168.2.99 `
  "cat /sys/class/fpga_manager/fpga0/state; systemctl is-enabled budgetcuda-fpga.service budgetcuda.service; systemctl is-active budgetcuda-fpga.service budgetcuda.service; ss -ltn | grep ':9000'"
```

Expected results are `operating`, two `enabled` lines, two `active` lines, and a
listener on `0.0.0.0:9000`.

## 8. Run board acceptance tests

Run the matched W8A8 and W4A8 graph smokes:

```powershell
& .\.venv\Scripts\python.exe FPGA\tests\hardware\pynq_transformer_graph_smoke.py `
  --address 192.168.2.99 --mode both --timeout 30
```

Run both HeatViT selector modes:

```powershell
& .\.venv\Scripts\python.exe FPGA\tests\hardware\pynq_heatvit_selector_smoke.py `
  --address 192.168.2.99 --mode both --timeout 30
```

Do not run direct MMIO, `/dev/mem`, XRT, or XLNK tests while the TCP daemon owns
the accelerator.

## 9. Run exact single-image diagnostics

Load the resident graph once, run diagnostic repetitions, and save the complete
module/GEMM cycle report:

```powershell
& .\.venv\Scripts\python.exe testing\deit_tiny\run_budgetcuda_deit.py `
  'testing\deit_tiny\data\cache_256\FreshApple\FreshApple (1).jpg' `
  --manifest testing\deit_tiny\artifacts\heatvit_w4a8_balanced\manifest.json `
  --address 192.168.2.99 `
  --port 9000 `
  --timeout 180 `
  --clock-hz 100000000 `
  --heatvit `
  --repeat 2 `
  --production-repeat 3 `
  --cycle-report testing\deit_tiny\results\fpga_cycle_report_current.json
```

The reference image must classify as `apple_fresh`. Record raw logits, total,
busy, and stall cycles, every module share, GEMM phase counters, clock, commit,
manifest, and release hashes.

## 10. Run the continuous resident-model benchmark

This benchmark performs one diagnostic graph, two warmups, and 20 production
graphs without reloading the model:

```powershell
& .\.venv\Scripts\python.exe testing\deit_tiny\run_continuous_benchmark.py `
  'testing\deit_tiny\data\cache_256\FreshApple\FreshApple (1).jpg' `
  --manifest testing\deit_tiny\artifacts\heatvit_w4a8_balanced\manifest.json `
  --address 192.168.2.99 `
  --port 9000 `
  --timeout 180 `
  --clock-hz 100000000 `
  --warmup 2 `
  --iterations 20 `
  --heatvit `
  --output testing\deit_tiny\results\fpga_continuous_benchmark.json
```

The verified redeployment run produced:

```text
mean host latency                         352.62 ms
mean device-graph time                   327.37 ms
sustained throughput                       2.835 images/s
device-graph-limited throughput            3.055 images/s
accelerator busy over stream wall time    77.68%
useful arithmetic vs wall-clock peak      19.98%
```

The current protocol serializes complete images. These are continuous
resident-model results, not cross-image FPGA overlap. Model upload took 16.63 s
and is excluded from steady-state inference performance.

## 11. USB/JTAG/UART recovery when networking is late

The PYNQ FT2232H appears as two direct FTDI links even when Windows creates no
COM port:

```text
1234-tulA / TUL A  JTAG
1234-tulB / TUL B  UART
```

Enumerate USB functions:

```powershell
pnputil.exe /enum-devices /connected /class USB
```

Enumerate the Zynq JTAG chain without programming it:

```powershell
& 'C:\Xilinx\Vivado\2024.2\bin\xsdb.bat' -eval `
  "connect; after 2000; puts [targets]; disconnect"
```

Expected targets include the APU, both Cortex-A9 cores, and `xc7z020`.

Read the UART directly through D2XX and request a prompt:

```powershell
& .\.venv\Scripts\python.exe testing\deit_tiny\scan_ftdi_uart.py `
  --serial 1234-tulB --seconds 10 --prompt
```

Query services even before SSH becomes available:

```powershell
& .\.venv\Scripts\python.exe testing\deit_tiny\scan_ftdi_uart.py `
  --serial 1234-tulB `
  --seconds 12 `
  --command "systemctl is-active ssh; systemctl is-active budgetcuda-fpga.service; systemctl is-active budgetcuda.service; cat /sys/class/fpga_manager/fpga0/state; ss -ltn"
```

A lit DONE LED confirms FPGA configuration but does not prove Linux services or
the accelerator daemon are ready. Wait for `operating`, active services, and
the port 9000 listener.

## 12. Recovery after execution status 7 or a wedged graph

First inspect service and FPGA state:

```powershell
ssh xilinx@192.168.2.99 `
  "systemctl --no-pager --full status budgetcuda.service budgetcuda-fpga.service; journalctl -u budgetcuda.service -n 100 --no-pager; cat /sys/class/fpga_manager/fpga0/state"
```

Restarting only the daemon clears resident allocations:

```powershell
ssh -t xilinx@192.168.2.99 "sudo systemctl restart budgetcuda.service"
```

If a graph returned firmware status 7 or the PL remains busy, reload the FPGA
before restarting the daemon:

```powershell
ssh -t xilinx@192.168.2.99 `
  "sudo systemctl stop budgetcuda.service; sudo systemctl restart budgetcuda-fpga.service; sudo systemctl start budgetcuda.service"
```

The host must reload the model after either recovery path.

## 13. Rollback and clean shutdown

The deployment script preserves the preceding installed release under:

```text
/opt/budgetcuda/backup-pre-heatvit/
```

Inspect that directory before attempting a rollback. Restore the daemon,
BIT/HWH pair, and both service units as one matched set; never roll back only a
single component.

Shut the board down cleanly before removing power:

```powershell
ssh -t xilinx@192.168.2.99 "sudo shutdown -h now"
```

## 14. Ground-zero acceptance checklist

- [ ] Clean `DeiT_Tiny` checkout and recorded commit.
- [ ] Complete model artifact directory and reference image restored.
- [ ] Host and firmware tests pass.
- [ ] Checked-in BIT/HWH hashes match, or rebuilt RTL passes all 27 benches and timing.
- [ ] Board has working compiler, PYNQ Python, FPGA manager, and XRT/ZOCL path.
- [ ] Board and host have stable Ethernet connectivity.
- [ ] ARM daemon is built from the same checkout as the host package.
- [ ] BIT, HWH, helper scripts, daemon, and systemd units are installed together.
- [ ] Both services are enabled and active; FPGA state is `operating`.
- [ ] W8A8/W4A8 graph and both HeatViT selector smokes pass.
- [ ] Exact diagnostic report preserves class/logits and includes all module/GEMM cycles.
- [ ] Continuous benchmark records latency, images/s, busy utilization, and arithmetic utilization.
- [ ] Commit, model mode, clock, and release hashes accompany the final record.
