#!/usr/bin/env python3
"""Read or set the PYNQ-Z2 PS FCLK0 divisor through the Zynq SLCR."""

from __future__ import annotations

import argparse
import mmap
import struct


SLCR_BASE = 0xF8000000
SLCR_UNLOCK = 0x08
SLCR_LOCK = 0x04
FCLK0_CTRL = 0x170
UNLOCK_KEY = 0x0000DF0D
LOCK_KEY = 0x0000767B

# The PYNQ-Z2 IO PLL runs at 1 GHz. FCLK0 divisor fields are DIVISOR1 in
# bits 25:20 and DIVISOR0 in bits 13:8.
FCLK0_SETTINGS = {
    25: 0x00201400,
    50: 0x00200A00,
    77: 0x00100D00,  # 1 GHz / 13 = 76.923 MHz
    90: 0x00100B00,  # 1 GHz / 11 = 90.909 MHz
    100: 0x00200500,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mhz", type=int, choices=tuple(FCLK0_SETTINGS))
    args = parser.parse_args()

    with open("/dev/mem", "r+b", buffering=0) as memory:
        registers = mmap.mmap(
            memory.fileno(), mmap.PAGESIZE, mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE, offset=SLCR_BASE,
        )
        try:
            before = struct.unpack_from("<I", registers, FCLK0_CTRL)[0]
            if args.mhz is not None:
                target = FCLK0_SETTINGS[args.mhz]
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
