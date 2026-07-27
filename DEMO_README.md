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
.\run_budgetcuda_demo.ps1
```

The script first runs a full 17-layer smoke comparison and then evaluates the
balanced 300-image FPGA subset. The validated release result is 99.0% FPGA
accuracy and 99.3333% agreement with ONNX Runtime.

The PYNQ 3.1.1 validation measured 270.714 ms mean and 285.892 ms p95 latency
over the same 300 images.

## Release fingerprints

- Model SHA-256:
  `a36f9701f126c6a76fb81366d4a384d9b920469749e5e32d2604e47e33a9b348`
- FPGA-manager payload SHA-256:
  `d91850f64f225f488f234606f8f3fa494b69e7d00c621134e7f56792d868ac6c`
- Routed timing: WNS `+0.089 ns`, TNS `0`, WHS `+0.019 ns` at 100 MHz.
