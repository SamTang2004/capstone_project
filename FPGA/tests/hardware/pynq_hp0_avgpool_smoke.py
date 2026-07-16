"""PYNQ-Z2 HP0 smoke test for AveragePool and GlobalAveragePool."""

from __future__ import annotations

import mmap
import os
import struct
import time

from pynq_hp0_cnn_smoke import PL_BASE, XlnkBuffer


CONTROL_START_AVGPOOL = 0x201


def main():
    fd = os.open("/dev/xlnk", os.O_RDWR)
    buffers = []
    try:
        source, output = XlnkBuffer(fd, 9), XlnkBuffer(fd, 4)
        buffers = [source, output]
        source.data[:] = b"\x00" * source.size
        output.data[:] = b"\x00" * output.size
        source.data[:9] = bytes((-7 & 0xFF, -2 & 0xFF, 3, 8, 4, -5 & 0xFF, 6, 9, -1 & 0xFF))

        page = mmap.PAGESIZE
        page_base = PL_BASE & ~(page - 1)
        page_offset = PL_BASE - page_base
        with open("/dev/mem", "r+b", buffering=0) as mem:
            regs = mmap.mmap(mem.fileno(), page_offset + 0x1000,
                             mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                             offset=page_base)
            try:
                def write(offset, value):
                    struct.pack_into("<I", regs, page_offset + offset, value & 0xFFFFFFFF)

                def read(offset):
                    return struct.unpack_from("<I", regs, page_offset + offset)[0]

                def run(channels, out_h, out_w, kernel_h, kernel_w, expected):
                    write(0x00, 0)
                    write(0x08, source.physical_address)
                    write(0x10, output.physical_address)
                    write(0x14, 1)             # N
                    write(0x18, channels)
                    write(0x1C, 3)             # input H
                    write(0x20, 3)             # input W
                    write(0x24, out_h)
                    write(0x28, out_w)
                    write(0x2C, kernel_h)
                    write(0x30, kernel_w)
                    write(0x34, 0x00000101)    # stride H=1, W=1
                    write(0x38, 0)             # no padding
                    write(0x00, CONTROL_START_AVGPOOL)
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        status = read(0x04)
                        if status & 4:
                            raise RuntimeError("AveragePool error status 0x%08x" % status)
                        if status & 2:
                            got = list(struct.unpack_from("<%db" % len(expected), output.data))
                            if got != expected:
                                raise RuntimeError("AveragePool mismatch: got %r expected %r" % (got, expected))
                            return got
                        time.sleep(0.0005)
                    raise TimeoutError("AveragePool timed out, status=0x%08x" % read(0x04))

                pooled = run(1, 2, 2, 2, 2, [0, 0, 6, 1])
                global_pooled = run(1, 1, 1, 3, 3, [1])
                source.data[:18] = bytes([3] * 9 + [18] * 9)
                two_channel_global = run(2, 1, 1, 3, 3, [3, 18])
                print("HP0 AveragePool smoke passed avg=%r global_avg=%r two_channel=%r" %
                      (pooled, global_pooled, two_channel_global))
            finally:
                regs.close()
    finally:
        for buf in reversed(buffers):
            buf.close()
        os.close(fd)


if __name__ == "__main__":
    main()
