"""Read a 32-bit physical register through /dev/mem on a PYNQ-Z2."""

from __future__ import annotations

import argparse
import mmap
import struct


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("address", type=lambda value: int(value, 0))
    args = parser.parse_args()

    page_base = args.address & ~(mmap.PAGESIZE - 1)
    page_offset = args.address - page_base
    with open("/dev/mem", "rb", buffering=0) as mem:
        registers = mmap.mmap(mem.fileno(), mmap.PAGESIZE, mmap.MAP_SHARED, mmap.PROT_READ, offset=page_base)
        try:
            value = struct.unpack_from("<I", registers, page_offset)[0]
        finally:
            registers.close()
    print(f"0x{args.address:08x}=0x{value:08x}")


if __name__ == "__main__":
    main()
