"""Safe on-board HP0 DDR smoke test using PYNQ-Z2's legacy Xlnk CMA driver."""

from __future__ import annotations

import fcntl
import mmap
import os
import random
import struct
import time
import argparse


PL_BASE = 0x43C80000
XLNK_DEVICE = "/dev/xlnk"
XLNK_IOCALLOCBUF = 0xC0045802  # _IOWR('X', 2, unsigned long) on 32-bit ARM.
TILE = 8
INT4_CONTROL = 0x100


def round_up_tile(value: int) -> int:
    return (value + TILE - 1) & ~(TILE - 1)


def write_int4(data: mmap.mmap, row_stride: int, row: int, col: int, value: int) -> None:
    offset = row * row_stride + col // 2
    nibble = value & 0xF
    current = data[offset]
    data[offset] = (
        (current & 0x0F) | (nibble << 4)
        if col & 1
        else (current & 0xF0) | nibble
    )


class XlnkBuffer:
    def __init__(self, fd: int, size: int):
        size = (size + mmap.PAGESIZE - 1) & ~(mmap.PAGESIZE - 1)
        request = bytearray(128)  # Larger than union xlnk_args on this kernel.
        struct.pack_into("<IiIB", request, 0, size, 0, 0, 0)
        fcntl.ioctl(fd, XLNK_IOCALLOCBUF, request, True)
        self.id, self.physical_address = struct.unpack_from("<iI", request, 4)
        if self.id <= 0 or self.physical_address == 0:
            raise RuntimeError("Xlnk CMA allocation failed")
        self.size = size
        self.data = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=self.id << 16)

    def close(self) -> None:
        self.data.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=TILE)
    parser.add_argument("--n", type=int, default=TILE)
    parser.add_argument("--k", type=int, default=TILE)
    parser.add_argument("--identity-b", action="store_true", help="Use a rectangular identity B for large, closed-form verification.")
    parser.add_argument("--seed", type=int, default=None, help="Use reproducible aperiodic INT8 A/B values generated from this seed.")
    parser.add_argument("--pad-to-tile", action="store_true", help="Zero-pad A/B/C and submit 16-aligned dimensions to hardware.")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()
    if min(args.m, args.n, args.k) <= 0:
        raise SystemExit("matrix dimensions must be positive")
    if args.timeout_seconds <= 0:
        raise SystemExit("timeout must be positive")
    hw_m = round_up_tile(args.m) if args.pad_to_tile else args.m
    hw_n = round_up_tile(args.n) if args.pad_to_tile else args.n
    hw_k = round_up_tile(args.k) if args.pad_to_tile else args.k
    a_stride = (hw_k + 1) // 2
    b_stride = (hw_n + 1) // 2

    fd = os.open(XLNK_DEVICE, os.O_RDWR)
    a = b = c = None
    page_size = mmap.PAGESIZE
    page_base = PL_BASE & ~(page_size - 1)
    page_offset = PL_BASE - page_base
    try:
        a = XlnkBuffer(fd, hw_m * a_stride)
        b = XlnkBuffer(fd, hw_k * b_stride)
        c = XlnkBuffer(fd, hw_m * hw_n * 4)
        print(
            f"allocated A=0x{a.physical_address:08x} B=0x{b.physical_address:08x} C=0x{c.physical_address:08x}",
            flush=True,
        )
        a.data[:] = b"\x00" * a.size
        b.data[:] = b"\x00" * b.size
        rng = random.Random(args.seed) if args.seed is not None else None
        a_values = [0] * (args.m * args.k)
        b_values = [0] * (args.k * args.n)
        for row in range(args.m):
            for col in range(args.k):
                value = rng.randint(-8, 7) if rng is not None else ((row * 3 + col * 5) % 16) - 8
                a_values[row * args.k + col] = value
                write_int4(a.data, a_stride, row, col, value)
        if args.identity_b:
            for index in range(min(args.k, args.n)):
                b_values[index * args.n + index] = 1
                write_int4(b.data, b_stride, index, index, 1)
        else:
            for row in range(args.k):
                for col in range(args.n):
                    value = rng.randint(-8, 7) if rng is not None else ((row * 7 + col * 2) % 13) - 6
                    b_values[row * args.n + col] = value
                    write_int4(b.data, b_stride, row, col, value)
        expected = None if args.identity_b else [
            sum(a_values[row * args.k + inner] * b_values[inner * args.n + col] for inner in range(args.k))
            for row in range(args.m) for col in range(args.n)
        ]
        c.data[:] = b"\x00" * c.size
        print("initialized CMA buffers", flush=True)

        print("opening register map", flush=True)
        with open("/dev/mem", "r+b", buffering=0) as mem:
            regs = mmap.mmap(mem.fileno(), page_offset + 0x1000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=page_base)
            try:
                print("register map open", flush=True)
                def write(offset: int, value: int) -> None:
                    struct.pack_into("<I", regs, page_offset + offset, value & 0xFFFFFFFF)

                def read(offset: int) -> int:
                    return struct.unpack_from("<I", regs, page_offset + offset)[0]

                print("clearing control", flush=True)
                write(0x00, 0)
                print("programming addresses", flush=True)
                write(0x08, a.physical_address)
                write(0x0C, b.physical_address)
                write(0x10, c.physical_address)
                write(0x14, a_stride)
                write(0x18, b_stride)
                write(0x1C, hw_n * 4)
                write(0x20, hw_m)
                write(0x24, hw_n)
                write(0x28, hw_k)
                write(0x2C, TILE)
                write(0x30, 2)
                print("starting control", flush=True)
                write(0x00, INT4_CONTROL | 1)
                print("submitted GEMM", flush=True)

                deadline = time.monotonic() + args.timeout_seconds
                while time.monotonic() < deadline:
                    status = read(0x04)
                    if status & 0x04:
                        raise RuntimeError(f"accelerator reported error status 0x{status:08x}")
                    # DONE can precede the last HP0 write response. Do not
                    # expose C until the common memory path is idle as well.
                    if status & 0x02 and not status & 0x01:
                        break
                    time.sleep(0.001)
                else:
                    raise TimeoutError(f"HP0 GEMM did not complete, final status=0x{read(0x04):08x} control=0x{read(0x00):08x}")
                print("GEMM completed", flush=True)
            finally:
                regs.close()

        print("verifying output", flush=True)
        first_row = [struct.unpack_from("<i", c.data, col * 4)[0] for col in range(min(hw_n, TILE))]
        if args.identity_b:
            first_expected = [a_values[col] if col < args.k else 0 for col in range(min(hw_n, TILE))]
        else:
            first_expected = expected[: min(args.n, TILE)]
        print(f"first output row actual={first_row} expected={first_expected}", flush=True)
        for row in range(args.m):
            for col in range(args.n):
                index = row * hw_n + col
                actual = struct.unpack_from("<i", c.data, index * 4)[0]
                if args.identity_b:
                    expected_value = a_values[row * args.k + col] if col < args.k else 0
                else:
                    expected_value = expected[row * args.n + col]
                if actual != expected_value:
                    raise RuntimeError(f"C[{row},{col}] expected {expected_value}, got {actual}")
        print(
            "HP0 DDR GEMM smoke passed "
            f"M={args.m} N={args.n} K={args.k} submitted={hw_m}x{hw_n}x{hw_k} "
            f"A=0x{a.physical_address:08x} B=0x{b.physical_address:08x} C=0x{c.physical_address:08x}"
        )
    finally:
        for buffer in (c, b, a):
            if buffer is not None:
                buffer.close()
        os.close(fd)


if __name__ == "__main__":
    main()
