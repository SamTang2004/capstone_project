#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -d "$script_dir/src" ] && [ -d "$script_dir/include" ]; then
    root=$script_dir
else
    root=$(CDPATH= cd -- "$script_dir/.." && pwd)
fi

xrt_cflags=
xrt_libs=
if [ -r /usr/include/xrt/xrt/xrt_bo.h ] && [ -e /usr/lib/libxrt_coreutil.so ]; then
    xrt_cflags="-DBCUDA_HAS_XRT=1 -I/usr/include/xrt"
    xrt_libs="-lxrt_coreutil"
fi

exec gcc -std=c11 -O2 -Wall -Wextra -I"$root/include" $xrt_cflags \
    "$root/src/protocol.c" "$root/src/buffer_manager.c" "$root/src/axi_dma.c" \
    "$root/src/graph_executor.c" "$root/src/command_dispatch.c" \
    "$root/src/protocol_server.c" "$root/src/budgetcuda_server.c" \
    $xrt_libs \
    -o "${1:-$root/budgetcuda-server}"
