"""PYNQ-Z2 HP0 smoke test for requantize and activation without GEMM."""

import mmap
import os
import struct
import time

from pynq_hp0_cnn_smoke import PL_BASE, XlnkBuffer


def main():
    fd = os.open("/dev/xlnk", os.O_RDWR)
    buffers = []
    try:
        source = XlnkBuffer(fd, 16)
        quant = XlnkBuffer(fd, 4)
        relu = XlnkBuffer(fd, 4)
        buffers = [source, quant, relu]
        for buf in buffers:
            buf.data[:] = b"\x00" * buf.size
        for index, value in enumerate((-4, -2, 2, 4)):
            struct.pack_into("<i", source.data, index * 4, value)

        page = mmap.PAGESIZE
        page_base = PL_BASE & ~(page - 1)
        page_offset = PL_BASE - page_base
        with open("/dev/mem", "r+b", buffering=0) as mem:
            regs = mmap.mmap(mem.fileno(), page_offset + 0x1000,
                             mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                             offset=page_base)
            try:
                def write(offset, value):
                    struct.pack_into("<I", regs, page_offset + offset,
                                     value & 0xFFFFFFFF)

                def read(offset):
                    return struct.unpack_from("<I", regs,
                                              page_offset + offset)[0]

                def run(control):
                    write(0x00, 0)
                    write(0x00, control)
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        status = read(0x04)
                        if status & 4:
                            raise RuntimeError("accelerator error 0x%08x" % status)
                        if status & 2:
                            return
                        time.sleep(0.0005)
                    raise TimeoutError("command 0x%x timed out, status=0x%08x" %
                                       (control, read(0x04)))

                write(0x08, source.physical_address)
                write(0x10, quant.physical_address)
                write(0x20, 4)
                write(0x34, 1)
                write(0x38, 0)
                write(0x3C, 0)
                run(0x3)
                got_quant = list(struct.unpack_from("<4b", quant.data))
                if got_quant != [-4, -2, 2, 4]:
                    raise RuntimeError("requantize mismatch: %r" % got_quant)

                write(0x08, quant.physical_address)
                write(0x10, relu.physical_address)
                write(0x20, 4)
                write(0x38, 0)
                run(0x9)
                got_relu = list(struct.unpack_from("<4b", relu.data))
                if got_relu != [0, 0, 2, 4]:
                    raise RuntimeError("ReLU mismatch: %r" % got_relu)
                print("HP0 post-op smoke passed requant=%r ReLU=%r" %
                      (got_quant, got_relu))
            finally:
                regs.close()
    finally:
        for buf in reversed(buffers):
            buf.close()
        os.close(fd)


if __name__ == "__main__":
    main()
