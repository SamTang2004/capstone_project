#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$script_dir/.." && pwd)
exec gcc -std=c11 -O2 -Wall -Wextra -Werror -I"$root/include" \
  "$root/src/protocol.c" "$root/src/buffer_manager.c" "$root/src/axi_dma.c" \
  "$root/src/graph_executor.c" "$root/src/command_dispatch.c" \
  "$root/src/protocol_server.c" "$root/src/budgetcuda_server.c" \
  -o "${1:-$root/budgetcuda-server}"
