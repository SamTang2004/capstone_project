"""Read or set the PYNQ-Z2 PS FCLK0 divisor through the Zynq SLCR.

The FPGA manager configures PL fabric only; it does not change the live PS
clock.  This helper is deliberately dependency-free so bring-up does not
depend on the board's PYNQ Python package.
"""

from __future__ import annotations

import argparse
import mmap
import os
import struct


SLCR_BASE = 0xF8000000
SLCR_UNLOCK = 0x08
SLCR_LOCK = 0x04
FCLK0_CTRL = 0x170
UNLOCK_KEY = 0x0000DF0D
LOCK_KEY = 0x0000767B
FCLK0_100_MHZ = 0x00200500
FCLK0_50_MHZ = 0x00200A00
FCLK0_25_MHZ = 0x00201400


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mhz", type=int, choices=(25, 50, 100))
    args = parser.parse_args()

    page_size = mmap.PAGESIZE
    with open("/dev/mem", "r+b", buffering=0) as mem:
        registers = mmap.mmap(mem.fileno(), page_size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=SLCR_BASE)
        try:
            before = struct.unpack_from("<I", registers, FCLK0_CTRL)[0]
            if args.mhz is not None:
                target = {25: FCLK0_25_MHZ, 50: FCLK0_50_MHZ, 100: FCLK0_100_MHZ}[args.mhz]
                struct.pack_into("<I", registers, SLCR_UNLOCK, UNLOCK_KEY)
                struct.pack_into("<I", registers, FCLK0_CTRL, target)
                struct.pack_into("<I", registers, SLCR_LOCK, LOCK_KEY)
            after = struct.unpack_from("<I", registers, FCLK0_CTRL)[0]
        finally:
            registers.close()

    print(f"fclk0_before=0x{before:08x} fclk0_after=0x{after:08x}")
    if args.mhz is not None and after != target:
        raise SystemExit("FCLK0 control register did not accept the requested divider")


if __name__ == "__main__":
    main()
