#!/bin/sh
set -eu

overlay_path=${BCUDA_OVERLAY_PATH:-/opt/budgetcuda/overlay/bcuda_pynq_z2.bit}
manager=/sys/class/fpga_manager/fpga0
clock_helper=/usr/local/lib/budgetcuda/pynq_fclk0.py
overlay_helper=/usr/local/lib/budgetcuda/budgetcuda_overlay.py
afi_helper=/usr/local/lib/budgetcuda/pynq_afi_hp0.py
pynq_python=/usr/local/share/pynq-venv/bin/python3
clock_mhz=${BCUDA_FCLK_MHZ:-100}

if [ ! -r "$overlay_path" ] || [ ! -r "${overlay_path%.bit}.hwh" ]; then
    echo "BudgetCUDA overlay pair is missing: $overlay_path" >&2
    exit 1
fi
if [ ! -x "$pynq_python" ] || [ ! -r "$clock_helper" ] || [ ! -r "$overlay_helper" ] || [ ! -r "$afi_helper" ]; then
    echo "BudgetCUDA PYNQ helpers are not installed" >&2
    exit 1
fi
if [ ! -w "$manager/firmware" ]; then
    echo "FPGA manager is unavailable: $manager/firmware" >&2
    exit 1
fi

# Stock PYNQ boot.py configures board services and then loads base.bin. Wait
# for that load so BudgetCUDA is always the final image programmed into PL.
if systemctl is-enabled bootpy.service >/dev/null 2>&1; then
    attempts=0
    until dmesg | grep -q "writing base.bin"; do
        attempts=$((attempts + 1))
        if [ "$attempts" -ge 90 ]; then
            echo "Timed out waiting for PYNQ boot.py to load base.bin" >&2
            exit 1
        fi
        /bin/sleep 2
    done
fi

/usr/bin/python3 "$afi_helper" --width 64
/usr/bin/python3 "$clock_helper" --mhz "$clock_mhz"
XILINX_XRT=/usr "$pynq_python" "$overlay_helper" "$overlay_path"

state=$(cat "$manager/state")
if [ "$state" != "operating" ]; then
    echo "BudgetCUDA FPGA programming failed: state=$state" >&2
    exit 1
fi

echo "BudgetCUDA FPGA is operating at ${clock_mhz} MHz with native 64-bit HP0 and $overlay_path"
