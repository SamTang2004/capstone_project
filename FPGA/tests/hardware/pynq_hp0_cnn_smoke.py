"""PYNQ-Z2 HP0 smoke test for biased Conv2D -> requantize -> ReLU."""

from __future__ import annotations

import fcntl
import mmap
import os
import struct
import time


PL_BASE = 0x43C80000
XLNK_IOCALLOCBUF = 0xC0045802


class XlnkBuffer:
    def __init__(self, fd: int, size: int):
        size = (size + mmap.PAGESIZE - 1) & ~(mmap.PAGESIZE - 1)
        request = bytearray(128)
        struct.pack_into("<IiIB", request, 0, size, 0, 0, 0)
        fcntl.ioctl(fd, XLNK_IOCALLOCBUF, request, True)
        self.id, self.physical_address = struct.unpack_from("<iI", request, 4)
        if self.id <= 0 or self.physical_address == 0:
            raise RuntimeError("Xlnk CMA allocation failed")
        self.size = size
        self.data = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=self.id << 16)

    def close(self):
        self.data.close()


def main():
    fd = os.open("/dev/xlnk", os.O_RDWR)
    buffers = []
    try:
        inp, weight, bias, conv, quant, relu = [XlnkBuffer(fd, size) for size in (9, 4, 4, 16, 4, 4)]
        buffers = [inp, weight, bias, conv, quant, relu]
        for buf in buffers:
            buf.data[:] = b"\x00" * buf.size
        inp.data[:9] = bytes(range(1, 10))
        weight.data[:4] = bytes((1, 0, 0, 1))
        struct.pack_into("<i", bias.data, 0, -10)

        page = mmap.PAGESIZE
        page_base = PL_BASE & ~(page - 1)
        page_offset = PL_BASE - page_base
        with open("/dev/mem", "r+b", buffering=0) as mem:
            regs = mmap.mmap(mem.fileno(), page_offset + 0x1000, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=page_base)
            try:
                def write(offset, value):
                    struct.pack_into("<I", regs, page_offset + offset, value & 0xFFFFFFFF)

                def read(offset):
                    return struct.unpack_from("<I", regs, page_offset + offset)[0]

                def run(control, timeout=10.0):
                    write(0x00, 0)
                    write(0x00, control)
                    deadline = time.monotonic() + timeout
                    while time.monotonic() < deadline:
                        status = read(0x04)
                        if status & 4:
                            raise RuntimeError("accelerator error status 0x%08x" % status)
                        if status & 2:
                            return
                        time.sleep(0.0005)
                    raise TimeoutError("accelerator command timed out")

                # Conv mode register overlay: C,H,W,outC,outH,outW,KH,KW,strides,pads,bias.
                write(0x08, inp.physical_address); write(0x0C, weight.physical_address); write(0x10, conv.physical_address)
                write(0x14, 1); write(0x18, 3); write(0x1C, 3)
                write(0x20, 1); write(0x24, 2); write(0x28, 2)
                write(0x2C, 2); write(0x30, 2); write(0x34, 0x00000101); write(0x38, 0)
                write(0x3C, bias.physical_address)
                run(0x5)
                got_conv = [struct.unpack_from("<i", conv.data, i * 4)[0] for i in range(4)]
                if got_conv != [-4, -2, 2, 4]:
                    raise RuntimeError("Conv2D mismatch: %r" % got_conv)

                write(0x08, conv.physical_address); write(0x10, quant.physical_address); write(0x20, 4)
                write(0x34, 1); write(0x38, 0); write(0x3C, 0)
                run(0x3)

                write(0x08, quant.physical_address); write(0x10, relu.physical_address); write(0x20, 4); write(0x38, 0)
                run(0x9)
                got_relu = [struct.unpack_from("<b", relu.data, i)[0] for i in range(4)]
                if got_relu != [0, 0, 2, 4]:
                    raise RuntimeError("ReLU mismatch: %r" % got_relu)
                print("HP0 CNN smoke passed Conv=%r ReLU=%r" % (got_conv, got_relu))
            finally:
                regs.close()
    finally:
        for buf in reversed(buffers):
            buf.close()
        os.close(fd)


if __name__ == "__main__":
    main()
