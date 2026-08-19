#!/usr/bin/env python3
"""Inspect or configure the Zynq-7000 HP0 AFI datapath width."""

from __future__ import annotations

import argparse
import ctypes
import mmap
import struct


AFI_HP0_BASE = 0xF8008000
AFI_MAP_SIZE = 0x1000
AFI_RDCHAN_CTRL = 0x00
AFI_WRCHAN_CTRL = 0x14
AFI_BUSWIDTH_MASK = 0x1


def read32(mapping: mmap.mmap, offset: int) -> int:
    return struct.unpack_from("<I", mapping, offset)[0]


def write32(mapping: mmap.mmap, offset: int, value: int) -> None:
    # AFI registers reject byte stores; ctypes emits one aligned 32-bit write,
    # matching the kernel driver's writel() operation.
    ctypes.c_uint32.from_buffer(mapping, offset).value = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, choices=(32, 64))
    args = parser.parse_args()

    with open("/dev/mem", "r+b", buffering=0) as memory:
        with mmap.mmap(memory.fileno(), AFI_MAP_SIZE, offset=AFI_HP0_BASE) as registers:
            before_read = read32(registers, AFI_RDCHAN_CTRL)
            before_write = read32(registers, AFI_WRCHAN_CTRL)
            if args.width is not None:
                width_bit = 1 if args.width == 32 else 0
                write32(registers, AFI_RDCHAN_CTRL, (before_read & ~AFI_BUSWIDTH_MASK) | width_bit)
                write32(registers, AFI_WRCHAN_CTRL, (before_write & ~AFI_BUSWIDTH_MASK) | width_bit)
            after_read = read32(registers, AFI_RDCHAN_CTRL)
            after_write = read32(registers, AFI_WRCHAN_CTRL)

    actual_width = 32 if (after_read & 1 and after_write & 1) else 64
    print(
        f"HP0 AFI read=0x{before_read:08x}->0x{after_read:08x} "
        f"write=0x{before_write:08x}->0x{after_write:08x} width={actual_width}"
    )
    if args.width is not None and actual_width != args.width:
        raise RuntimeError(f"failed to configure HP0 AFI for {args.width}-bit operation")


if __name__ == "__main__":
    main()
