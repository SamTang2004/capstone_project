"""Dependency-free AXI-Lite smoke test for the PYNQ-Z2 Budget-CUDA overlay."""

from __future__ import annotations

import argparse
import mmap
import os
import struct


PL_BASE = 0x43C80000
PL_CONTROL = 0x00
PL_STATUS = 0x04
PL_STATUS_BUSY = 0x01
PL_STATUS_ERROR = 0x04


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=lambda value: int(value, 0), default=PL_BASE)
    args = parser.parse_args()

    page_size = mmap.PAGESIZE
    page_base = args.base & ~(page_size - 1)
    page_offset = args.base - page_base
    with open("/dev/mem", "r+b", buffering=0) as mem:
        registers = mmap.mmap(mem.fileno(), page_offset + 0x1000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=page_base)
        try:
            before = struct.unpack_from("<I", registers, page_offset + PL_STATUS)[0]
            struct.pack_into("<I", registers, page_offset + PL_CONTROL, 0)
            after = struct.unpack_from("<I", registers, page_offset + PL_STATUS)[0]
        finally:
            registers.close()

    print(f"status_before=0x{before:08x} status_after=0x{after:08x}")
    if after & (PL_STATUS_BUSY | PL_STATUS_ERROR):
        raise SystemExit("accelerator did not enter an idle, non-error state")


if __name__ == "__main__":
    main()
