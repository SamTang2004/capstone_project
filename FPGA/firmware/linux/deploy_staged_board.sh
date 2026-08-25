#!/bin/sh
set -eu

stage=${1:-/tmp/bcuda-deploy}
backup=/opt/budgetcuda/backup-pre-heatvit

systemctl stop budgetcuda.service budgetcuda-fpga.service
install -d "$backup" /opt/budgetcuda/overlay /usr/local/lib/budgetcuda

for path in \
    /opt/budgetcuda/budgetcuda-server \
    /opt/budgetcuda/overlay/bcuda_pynq_z2.bit \
    /opt/budgetcuda/overlay/bcuda_pynq_z2.hwh \
    /etc/systemd/system/budgetcuda.service \
    /etc/systemd/system/budgetcuda-fpga.service; do
    if [ -e "$path" ]; then
        cp -a "$path" "$backup/"
    fi
done

install -m 0755 "$stage/budgetcuda-server" /opt/budgetcuda/budgetcuda-server
install -m 0644 "$stage/bcuda_pynq_z2.bit" /opt/budgetcuda/overlay/bcuda_pynq_z2.bit
install -m 0644 "$stage/bcuda_pynq_z2.hwh" /opt/budgetcuda/overlay/bcuda_pynq_z2.hwh
install -m 0755 "$stage/budgetcuda-fpga-setup.sh" /usr/local/sbin/budgetcuda-fpga-setup
install -m 0644 "$stage/budgetcuda_overlay.py" /usr/local/lib/budgetcuda/budgetcuda_overlay.py
install -m 0644 "$stage/pynq_fclk0.py" /usr/local/lib/budgetcuda/pynq_fclk0.py
install -m 0644 "$stage/pynq_afi_hp0.py" /usr/local/lib/budgetcuda/pynq_afi_hp0.py
install -m 0644 "$stage/budgetcuda-fpga.service" /etc/systemd/system/budgetcuda-fpga.service
install -m 0644 "$stage/budgetcuda.service" /etc/systemd/system/budgetcuda.service

systemctl daemon-reload
systemctl reset-failed budgetcuda-fpga.service budgetcuda.service
systemctl start budgetcuda-fpga.service
systemctl start budgetcuda.service
systemctl --no-pager --full status budgetcuda-fpga.service budgetcuda.service
