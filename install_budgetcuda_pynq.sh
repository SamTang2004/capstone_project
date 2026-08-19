#!/bin/sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "Run this installer as root: sudo ./install_budgetcuda_pynq.sh" >&2
    exit 1
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
fpga_dir=$script_dir/FPGA
linux_dir=$fpga_dir/firmware/linux
payload=$fpga_dir/hardware/bcuda_pynq_z2_fpga_manager.bin
bitfile=$fpga_dir/hardware/bcuda_pynq_z2.bit
hwh=$fpga_dir/hardware/bcuda_pynq_z2.hwh
clock_helper=$fpga_dir/tests/hardware/pynq_fclk0.py

for required in \
    "$payload" \
    "$bitfile" \
    "$hwh" \
    "$clock_helper" \
    "$linux_dir/build_server.sh" \
    "$linux_dir/budgetcuda-fpga-setup.sh" \
    "$linux_dir/budgetcuda_overlay.py" \
    "$linux_dir/pynq_afi_hp0.py" \
    "$linux_dir/budgetcuda-fpga.service" \
    "$linux_dir/budgetcuda.service"
do
    if [ ! -f "$required" ]; then
        echo "Required release file is missing: $required" >&2
        exit 1
    fi
done

build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT HUP INT TERM

echo "Building the BudgetCUDA ARM daemon..."
sh "$linux_dir/build_server.sh" "$build_dir/budgetcuda-server"

echo "Installing the daemon, FPGA payload, and boot services..."
systemctl stop budgetcuda.service 2>/dev/null || true
# Stock boot.py also participates in PYNQ board/network initialization. Keep it
# enabled; the FPGA helper waits for its base.bin load before programming the
# final BudgetCUDA image.
systemctl unmask bootpy.service 2>/dev/null || true
systemctl enable bootpy.service
install -d /opt/budgetcuda/overlay /usr/local/lib/budgetcuda
install -m 0755 "$build_dir/budgetcuda-server" /opt/budgetcuda/budgetcuda-server
install -m 0644 "$payload" /lib/firmware/bcuda_pynq_z2_fpga_manager.bin
install -m 0644 "$bitfile" /opt/budgetcuda/overlay/bcuda_pynq_z2.bit
install -m 0644 "$hwh" /opt/budgetcuda/overlay/bcuda_pynq_z2.hwh
install -m 0644 "$clock_helper" /usr/local/lib/budgetcuda/pynq_fclk0.py
install -m 0644 "$linux_dir/budgetcuda_overlay.py" /usr/local/lib/budgetcuda/budgetcuda_overlay.py
install -m 0644 "$linux_dir/pynq_afi_hp0.py" /usr/local/lib/budgetcuda/pynq_afi_hp0.py
install -m 0755 "$linux_dir/budgetcuda-fpga-setup.sh" /usr/local/sbin/budgetcuda-fpga-setup
install -m 0644 "$linux_dir/budgetcuda-fpga.service" /etc/systemd/system/budgetcuda-fpga.service
install -m 0644 "$linux_dir/budgetcuda.service" /etc/systemd/system/budgetcuda.service

systemctl daemon-reload
systemctl enable budgetcuda-fpga.service budgetcuda.service
systemctl restart budgetcuda-fpga.service
systemctl restart budgetcuda.service

test "$(cat /sys/class/fpga_manager/fpga0/state)" = "operating"
test "$(systemctl is-active budgetcuda-fpga.service)" = "active"
test "$(systemctl is-active budgetcuda.service)" = "active"

echo "BudgetCUDA installation complete."
echo "FPGA state: $(cat /sys/class/fpga_manager/fpga0/state)"
echo "Daemon: $(systemctl is-active budgetcuda.service)"
echo "PYNQ boot initialization: $(systemctl is-enabled bootpy.service 2>/dev/null || true)"
