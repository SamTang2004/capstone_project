#!/usr/bin/env python3
"""Read 32-bit physical words for PYNQ address-path diagnostics."""

from __future__ import annotations

import argparse
import mmap
import struct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("addresses", nargs="+", type=lambda value: int(value, 0))
    args = parser.parse_args()

    with open("/dev/mem", "rb", buffering=0) as memory:
        for address in args.addresses:
            page_base = address & ~(mmap.PAGESIZE - 1)
            page_offset = address - page_base
            mapping = mmap.mmap(
                memory.fileno(), mmap.PAGESIZE, mmap.MAP_SHARED,
                mmap.PROT_READ, offset=page_base,
            )
            try:
                value = struct.unpack_from("<I", mapping, page_offset)[0]
            finally:
                mapping.close()
            print(f"0x{address:08x}: 0x{value:08x}")


if __name__ == "__main__":
    main()
