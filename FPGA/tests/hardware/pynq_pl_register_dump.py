"""Dump the BudgetCUDA PL control and Conv diagnostics register block."""

from __future__ import annotations

import argparse
import mmap
import os
import struct


REGISTER_NAMES = (
    "control", "status", "a_base", "b_base", "c_base", "a_stride", "b_stride", "c_stride",
    "m", "n", "k", "tile_size", "array_count", "rq_multiplier", "rq_right_shift", "rq_zero_point",
    "scheduler_state", "memory_state", "reserved_18", "reserved_19", "reserved_20", "reserved_21",
    "reserved_22", "reserved_23", "reserved_24", "reserved_25", "hp0_read_shape",
    "hp0_read_response", "cache_hits", "cache_misses", "fill_cycles", "bytes_delivered",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=lambda value: int(value, 0), default=0x43C80000)
    args = parser.parse_args()

    page_size = mmap.PAGESIZE
    page_base = args.base & ~(page_size - 1)
    page_offset = args.base - page_base
    fd = os.open("/dev/mem", os.O_RDONLY | os.O_SYNC)
    try:
        with mmap.mmap(fd, page_size, access=mmap.ACCESS_READ, offset=page_base) as registers:
            status = 0
            for index, name in enumerate(REGISTER_NAMES):
                offset = index * 4
                value = struct.unpack_from("<I", registers, page_offset + offset)[0]
                if index == 1:
                    status = value
                print(f"0x{offset:02x} {name:20s} 0x{value:08x} ({value})")
            print(
                "HP0 read telemetry: "
                f"ARLEN={status >> 28 & 0xF} "
                f"ARSIZE={status >> 25 & 0x7} "
                f"RBEATS={status >> 20 & 0x1F} "
                f"RRESP={status >> 18 & 0x3}"
            )
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
