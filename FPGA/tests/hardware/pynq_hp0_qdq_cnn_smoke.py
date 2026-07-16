"""PYNQ-Z2 HP0 smoke test for a chained quantized CNN inference path."""

from __future__ import annotations

import mmap
import os
import struct
import time

from pynq_hp0_cnn_smoke import PL_BASE, XlnkBuffer


CONTROL_CONV = 0x5
CONTROL_REQUANTIZE = 0x3
CONTROL_RELU = 0x9
CONTROL_GLOBAL_AVGPOOL = 0x201


def main():
    fd = os.open("/dev/xlnk", os.O_RDWR)
    buffers = []
    try:
        sizes = (9, 2, 8, 72, 18, 18, 2, 6, 12, 12, 3)
        (source, conv_weight, conv_bias, conv_acc, conv_quant, relu,
         pooled, classifier_weight, classifier_bias, classifier_acc,
         logits) = [XlnkBuffer(fd, size) for size in sizes]
        buffers = [source, conv_weight, conv_bias, conv_acc, conv_quant, relu,
                   pooled, classifier_weight, classifier_bias, classifier_acc,
                   logits]
        for buf in buffers:
            buf.data[:] = b"\x00" * buf.size

        source.data[:9] = bytes((-7 & 0xFF, -2 & 0xFF, 3, 8, 4, -5 & 0xFF, 6, 9, -1 & 0xFF))
        conv_weight.data[:2] = bytes((1, -1 & 0xFF))
        struct.pack_into("<2i", conv_bias.data, 0, 0, 20)
        classifier_weight.data[:6] = bytes((1, 1, 1, -1 & 0xFF, -1 & 0xFF, 1))
        struct.pack_into("<3i", classifier_bias.data, 0, 0, 0, 0)

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

                def run(control):
                    write(0x00, 0)
                    write(0x00, control)
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        status = read(0x04)
                        if status & 4:
                            raise RuntimeError("PL error status 0x%08x for control 0x%x" % (status, control))
                        # DONE marks the last producer request, while BUSY
                        # remains set until its HP0 write is committed.
                        if status & 2 and not status & 1:
                            return
                        time.sleep(0.0005)
                    raise TimeoutError("PL command 0x%x timed out, status=0x%08x" % (control, read(0x04)))

                def conv(inp, weight, bias, output, in_c, in_h, in_w, out_c):
                    write(0x08, inp.physical_address); write(0x0C, weight.physical_address); write(0x10, output.physical_address)
                    write(0x14, in_c); write(0x18, in_h); write(0x1C, in_w)
                    write(0x20, out_c); write(0x24, 1 if in_h == 1 else in_h); write(0x28, 1 if in_w == 1 else in_w)
                    write(0x2C, 1); write(0x30, 1); write(0x34, 0x00000101); write(0x38, 0); write(0x3C, bias.physical_address)
                    run(CONTROL_CONV)

                def requantize(inp, output, count):
                    write(0x08, inp.physical_address); write(0x10, output.physical_address); write(0x20, count)
                    write(0x34, 1); write(0x38, 0); write(0x3C, 0)
                    run(CONTROL_REQUANTIZE)

                conv(source, conv_weight, conv_bias, conv_acc, 1, 3, 3, 2)
                requantize(conv_acc, conv_quant, 18)
                write(0x08, conv_quant.physical_address); write(0x10, relu.physical_address); write(0x20, 18); write(0x38, 0)
                run(CONTROL_RELU)

                # GlobalAveragePool N=1, C=2, H=W=3, kernel=3, stride=1.
                write(0x08, relu.physical_address); write(0x10, pooled.physical_address)
                write(0x14, 1); write(0x18, 2); write(0x1C, 3); write(0x20, 3); write(0x24, 1); write(0x28, 1)
                write(0x2C, 3); write(0x30, 3); write(0x34, 0x00000101); write(0x38, 0)
                run(CONTROL_GLOBAL_AVGPOOL)
                got_pool = list(struct.unpack_from("<2b", pooled.data))
                if got_pool != [3, 18]:
                    raise RuntimeError("GlobalAveragePool mismatch: %r" % got_pool)

                conv(pooled, classifier_weight, classifier_bias, classifier_acc, 2, 1, 1, 3)
                requantize(classifier_acc, logits, 3)
                got_logits = list(struct.unpack_from("<3b", logits.data))
                if got_logits != [21, -15, 15]:
                    raise RuntimeError("Classifier mismatch: %r" % got_logits)
                print("HP0 Q/DQ CNN smoke passed pool=%r logits=%r" % (got_pool, got_logits))
            finally:
                regs.close()
    finally:
        for buf in reversed(buffers):
            buf.close()
        os.close(fd)


if __name__ == "__main__":
    main()
