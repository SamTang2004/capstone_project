"""Read a small physical-memory range from the PYNQ PS for HP0 diagnosis."""

from __future__ import annotations

import argparse
import mmap
import os


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("address", type=lambda value: int(value, 0))
    parser.add_argument("--bytes", type=int, default=64)
    args = parser.parse_args()

    if args.bytes <= 0 or args.bytes > mmap.PAGESIZE:
        raise ValueError("--bytes must be between 1 and one page")

    page_base = args.address & ~(mmap.PAGESIZE - 1)
    page_offset = args.address - page_base
    if page_offset + args.bytes > mmap.PAGESIZE:
        raise ValueError("requested range crosses a page boundary")

    fd = os.open("/dev/mem", os.O_RDONLY | os.O_SYNC)
    try:
        mapping = mmap.mmap(
            fd,
            mmap.PAGESIZE,
            mmap.MAP_SHARED,
            mmap.PROT_READ,
            offset=page_base,
        )
        try:
            print(mapping[page_offset : page_offset + args.bytes].hex())
        finally:
            mapping.close()
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
