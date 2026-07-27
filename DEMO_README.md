# BudgetCUDA PYNQ-Z2 Demo

This release contains the PYNQ-Z2 INT8 bitstream, matching PS daemon source,
the 99% fruit-classification ONNX model, and the deterministic 300-image
Fruits-360 test subset used for the presentation.

## Install the board

Clone this branch on the PYNQ-Z2 and run from the repository root:

```sh
sudo ./install_budgetcuda_pynq.sh
```

The installer builds the ARM daemon on the board, installs the paired `.bit`
and `.hwh` overlay, sets FCLK0 to 100 MHz, and enables both services. On every
later boot, `budgetcuda-fpga.service` loads the overlay before
`budgetcuda.service` starts.
PYNQ's stock `bootpy.service` remains enabled because it participates in board
and network initialization. The FPGA helper waits until boot.py has loaded
`base.bin`, then loads BudgetCUDA through the PYNQ Overlay API so it is the
final PL image and its PSDDR metadata is registered with XRT. The daemon
reserves and retains a 16 MiB XRT arena. Older PYNQ images retain the legacy
Xlnk allocator fallback.

Verify a cold boot with:

```sh
cat /sys/class/fpga_manager/fpga0/state
systemctl is-active budgetcuda-fpga.service
systemctl is-active budgetcuda.service
systemctl is-enabled bootpy.service
ss -ltn | grep ':9000'
```

The expected values are `operating`, `active`, `active`, `enabled`, and a
listener on TCP port 9000.

## Run the host demo

The Windows host requires the repository `.venv` with NumPy, Pillow, ONNX, and
ONNX Runtime installed. Connect the host at `192.168.2.1/24` directly to the
board at `192.168.2.99`, then run from the repository root:

```powershell
.\install_requirements.bat
```

This installs the validated non-Torch dependencies from `requirements.txt`.
Torch and torchvision are deliberately excluded and must be installed
separately using builds that match the host CUDA version.

Then run:

```powershell
.\run_budgetcuda_demo.ps1
```

The script first runs a full 17-layer smoke comparison and then evaluates the
balanced 300-image FPGA subset. The validated release result is 99.0% FPGA
accuracy and 99.3333% agreement with ONNX Runtime.

The PYNQ 3.1.1 validation measured 270.714 ms mean and 285.892 ms p95 latency
over the same 300 images.

## Demo action sequence

Run these steps from the repository root on the Windows host.

### 1. Start and wait for the board

1. Connect the PYNQ-Z2 to Ethernet and power.
2. Confirm the boot jumper is set to SD.
3. Perform a manual power cycle.
4. Allow approximately 2.5 minutes for PYNQ and BudgetCUDA to start.

Check network and service-port readiness:

```powershell
Test-Connection 192.168.2.99 -Count 2
Test-NetConnection 192.168.2.99 -Port 9000
```

Continue when `TcpTestSucceeded` is `True`.

### 2. Verify the board services

Connect to the board. The demo image uses the username `xilinx` and password
`xilinx`.

```powershell
ssh xilinx@192.168.2.99
```

Run these commands on the board:

```sh
cat /sys/class/fpga_manager/fpga0/state
systemctl is-active budgetcuda-fpga.service
systemctl is-active budgetcuda.service
ss -ltn | grep ':9000'
sudo journalctl -b -u budgetcuda-fpga.service -u budgetcuda.service --no-pager
```

Expected evidence:

- FPGA state is `operating`.
- Both BudgetCUDA services are `active`.
- TCP port `9000` is listening.
- The log reports `PSDDR base=0x0 size=268435456`.
- The log reports a 16 MiB XRT arena with a 32-bit physical address.
- FCLK0 reports `0x00200500`, corresponding to 100 MHz.

Exit the SSH session before running host tests:

```sh
exit
```

### 3. Run the basic accelerator smoke test

This uploads a small model descriptor and five tensors, executes it through the
board service, and checks the returned values.

```powershell
.\.venv\Scripts\python.exe FPGA\tests\hardware\pynq_model_load_smoke.py `
    --address 192.168.2.99 --port 9000 --timeout 60
```

Expected result:

```text
model load smoke passed: 5 tensors and descriptor uploaded; output=[0, 0, 2, 4]
```

### 4. Run one complete CNN inference

This imports the tracked ONNX model, uploads all 17 layers, executes the graph
on the FPGA, and compares the result with ONNX Runtime and the virtual PL.

```powershell
.\.venv\Scripts\python.exe FPGA\tests\hardware\pynq_full_cnn_inference.py `
    --model FPGA\demo\models\fruit3_budgetcuda_int8.onnx `
    --address 192.168.2.99 --port 9000 --timeout 300
```

The output must begin with `Full CNN inference passed`. The validated run used
17 layers, 30 buffers, and 560818 bytes, and selected class `1`.

### 5. Run the optional accuracy and latency benchmark

This runs 100 apple, 100 banana, and 100 orange images. It normally takes about
90 seconds.

```powershell
.\.venv\Scripts\python.exe FPGA\tests\hardware\pynq_fruit_accuracy.py `
    --model FPGA\demo\models\fruit3_budgetcuda_int8.onnx `
    --dataset-root FPGA\demo\dataset\fruits-360-100x100 `
    --address 192.168.2.99 --port 9000 --timeout 300 `
    --skip-train --board-per-class 100 --seed 2026
```

Validated presentation result:

```text
FPGA balanced-test accuracy=0.990000 images=300 onnx_agreement=0.993333
mean_ms=270.714 p95_ms=285.892 max_ms=289.849
```

For the complete sequence in one command, run:

```powershell
.\run_budgetcuda_demo.ps1
```

### Recovery during the demo

If SSH works but port `9000` does not, inspect the services first:

```sh
sudo systemctl status budgetcuda-fpga.service budgetcuda.service --no-pager -l
sudo journalctl -b -u budgetcuda-fpga.service -u budgetcuda.service --no-pager
```

Restart them in dependency order:

```sh
sudo systemctl stop budgetcuda.service
sudo systemctl restart budgetcuda-fpga.service
sudo systemctl start budgetcuda.service
```

If Ethernet does not return after a software reboot, perform a manual power
cycle and wait approximately 2.5 minutes. Do not manually program another
bitstream after the services become active because that would replace the
BudgetCUDA overlay and invalidate its XRT memory metadata.

## Release fingerprints

- Model SHA-256:
  `a36f9701f126c6a76fb81366d4a384d9b920469749e5e32d2604e47e33a9b348`
- FPGA-manager payload SHA-256:
  `d91850f64f225f488f234606f8f3fa494b69e7d00c621134e7f56792d868ac6c`
- Routed timing: WNS `+0.089 ns`, TNS `0`, WHS `+0.019 ns` at 100 MHz.
