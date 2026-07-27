#!/bin/sh
set -eu

firmware_name=${BCUDA_FIRMWARE_NAME:-bcuda_pynq_z2_fpga_manager.bin}
firmware_path=/lib/firmware/$firmware_name
manager=/sys/class/fpga_manager/fpga0
clock_helper=/usr/local/lib/budgetcuda/pynq_fclk0.py

if [ ! -r "$firmware_path" ]; then
    echo "BudgetCUDA FPGA payload is missing: $firmware_path" >&2
    exit 1
fi
if [ ! -x /usr/bin/python3 ] || [ ! -r "$clock_helper" ]; then
    echo "BudgetCUDA FCLK0 helper is not installed: $clock_helper" >&2
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

/usr/bin/python3 "$clock_helper" --mhz 100
printf '%s\n' "$firmware_name" > "$manager/firmware"
/bin/sleep 1

state=$(cat "$manager/state")
if [ "$state" != "operating" ]; then
    echo "BudgetCUDA FPGA programming failed: state=$state" >&2
    exit 1
fi

echo "BudgetCUDA FPGA is operating at 100 MHz with $firmware_name"
