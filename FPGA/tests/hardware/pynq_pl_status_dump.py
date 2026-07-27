"""Dump BudgetCUDA PL status and diagnostic counters through direct MMIO."""

from __future__ import annotations

import argparse
import mmap
import struct


PL_BASE = 0x43C80000
DIAGNOSTIC_NAMES = (
    "total_cycles",
    "engine_cycles",
    "compute_cycles",
    "load_cycles",
    "cache_wait_cycles",
    "cache_ready_cycles",
    "ddr_read_commands",
    "ddr_read_beats",
    "ddr_write_commands",
    "ddr_stall_cycles",
    "core_rows",
    "output_stall_cycles",
    "descriptor_count",
    "cache_full_cycles",
    "reserved_14",
    "reserved_15",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=lambda value: int(value, 0), default=PL_BASE)
    args = parser.parse_args()

    page_size = mmap.PAGESIZE
    page_base = args.base & ~(page_size - 1)
    page_offset = args.base - page_base
    with open("/dev/mem", "r+b", buffering=0) as memory:
        registers = mmap.mmap(
            memory.fileno(),
            page_offset + page_size,
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE,
            offset=page_base,
        )
        try:
            def read(offset: int) -> int:
                return struct.unpack_from("<I", registers, page_offset + offset)[0]

            status = read(0x04)
            print(
                f"status=0x{status:08x} active={status & 1} "
                f"done={(status >> 1) & 1} error={(status >> 2) & 1} "
                f"load_state={(status >> 3) & 0xf} "
                f"engine_busy={(status >> 7) & 1} "
                f"lookahead_busy={(status >> 8) & 1} "
                f"conv_loader_busy={(status >> 9) & 1} "
                f"writer_valid={(status >> 10) & 1} "
                f"memio_busy={(status >> 11) & 1} "
                f"mem_cmd_valid={(status >> 12) & 1} "
                f"mem_cmd_ready={(status >> 13) & 1}"
            )
            error_sources = (
                ("engine", 16),
                ("selected_loader", 17),
                ("conv_loader", 18),
                ("row_stream", 19),
                ("bias", 20),
                ("requantize", 21),
                ("activation", 22),
                ("tensor", 23),
                ("memory_io", 24),
                ("load_buffer", 25),
                ("lookahead", 26),
                ("invalid_mode", 27),
            )
            asserted = [
                name for name, bit in error_sources if status & (1 << bit)
            ]
            print(
                "error_sources="
                + (",".join(asserted) if asserted else "none")
            )
            for index, name in enumerate(DIAGNOSTIC_NAMES):
                print(f"{name}={read(0x40 + index * 4)}")
        finally:
            registers.close()


if __name__ == "__main__":
    main()
