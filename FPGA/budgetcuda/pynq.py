from __future__ import annotations

from dataclasses import dataclass
import mmap
import os
import struct

from .compiler import ModelPackage, upload_model
from .device import Device
from .dtypes import DType, dtype_size, normalize_dtype


PL_REGISTER_BASE = 0x43C80000
PL_REGISTER_RANGE = 0x10000
PL_CONTROL = 0x00
PL_STATUS = 0x04
PL_A_BASE = 0x08
PL_B_BASE = 0x0C
PL_C_BASE = 0x10
PL_A_STRIDE = 0x14
PL_B_STRIDE = 0x18
PL_C_STRIDE = 0x1C
PL_M = 0x20
PL_N = 0x24
PL_K = 0x28
PL_TILE_SIZE = 0x2C
PL_ARRAY_COUNT = 0x30
PL_START = 0x1
PL_STATUS_DONE = 0x2
PL_STATUS_ERROR = 0x4

XLNK_DEVICE = "/dev/xlnk"
XLNK_IOCALLOCBUF = 0xC0045802  # _IOWR('X', 2, unsigned long) on 32-bit ARM.
SLCR_BASE = 0xF8000000
SLCR_UNLOCK = 0x08
SLCR_LOCK = 0x04
FCLK0_CTRL = 0x170
SLCR_UNLOCK_KEY = 0x0000DF0D
SLCR_LOCK_KEY = 0x0000767B
FCLK0_VALUES = {50: 0x00200A00, 100: 0x00200500}


@dataclass
class PynqBuffer:
    data: object
    dtype: DType
    nbytes: int

    @property
    def physical_address(self) -> int:
        return int(self.data.physical_address)

    def flush(self) -> None:
        self.data.flush()

    def invalidate(self) -> None:
        self.data.invalidate()

    def close(self) -> None:
        close = getattr(self.data, "close", None)
        if close is not None:
            close()


class LegacyXlnkBuffer:
    """Non-cacheable CMA buffer backed by the PYNQ-Z2 legacy Xlnk driver."""

    def __init__(self, fd: int, nbytes: int):
        import fcntl

        nbytes = (nbytes + mmap.PAGESIZE - 1) & ~(mmap.PAGESIZE - 1)
        request = bytearray(128)
        struct.pack_into("<IiIB", request, 0, nbytes, 0, 0, 0)
        fcntl.ioctl(fd, XLNK_IOCALLOCBUF, request, True)
        self._id, self.physical_address = struct.unpack_from("<iI", request, 4)
        if self._id <= 0 or self.physical_address == 0:
            raise RuntimeError("Xlnk CMA allocation failed")
        self.nbytes = nbytes
        self.data = mmap.mmap(fd, nbytes, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=self._id << 16)

    def flush(self) -> None:
        # The allocation is non-cacheable, so CPU and PL share the same view.
        return None

    def invalidate(self) -> None:
        return None

    def close(self) -> None:
        self.data.close()


class LegacyXlnkAllocator:
    """Allocator adapter matching PYNQ's ``allocate(shape, dtype)`` API."""

    _DTYPE_BYTES = {"int8": 1, "int16": 2, "int32": 4}

    def __init__(self, device: str = XLNK_DEVICE):
        self._fd = os.open(device, os.O_RDWR)

    def __call__(self, *, shape, dtype: str) -> LegacyXlnkBuffer:
        try:
            item_bytes = self._DTYPE_BYTES[dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported Xlnk buffer dtype: {dtype}") from exc
        count = 1
        for dimension in shape:
            count *= int(dimension)
        if count <= 0:
            raise ValueError("Xlnk buffer shape must contain at least one element")
        return LegacyXlnkBuffer(self._fd, count * item_bytes)

    def close(self) -> None:
        os.close(self._fd)


class DirectMmio:
    """Dependency-free AXI-Lite MMIO accessor for a PYNQ-Z2 Linux image."""

    def __init__(self, base: int, length: int):
        page_size = mmap.PAGESIZE
        self._offset = base & (page_size - 1)
        self._mem = open("/dev/mem", "r+b", buffering=0)
        self._map = mmap.mmap(
            self._mem.fileno(), self._offset + length, mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE, offset=base - self._offset,
        )

    def write(self, offset: int, value: int) -> None:
        struct.pack_into("<I", self._map, self._offset + offset, value & 0xFFFFFFFF)

    def read(self, offset: int) -> int:
        return struct.unpack_from("<I", self._map, self._offset + offset)[0]

    def close(self) -> None:
        self._map.close()
        self._mem.close()


def configure_pynq_z2_fclk0(mhz: int = 50) -> None:
    """Set the live PYNQ-Z2 FCLK0 divider before programming a PL image.

    FPGA manager reconfiguration does not update PS clock dividers. The
    accelerator bitstream must be programmed only after this function selects
    the frequency used during implementation.
    """

    try:
        value = FCLK0_VALUES[mhz]
    except KeyError as exc:
        raise ValueError("Supported PYNQ-Z2 FCLK0 rates are 50 and 100 MHz") from exc
    slcr = DirectMmio(SLCR_BASE, mmap.PAGESIZE)
    try:
        slcr.write(SLCR_UNLOCK, SLCR_UNLOCK_KEY)
        slcr.write(FCLK0_CTRL, value)
        slcr.write(SLCR_LOCK, SLCR_LOCK_KEY)
        if slcr.read(FCLK0_CTRL) != value:
            raise RuntimeError("PYNQ-Z2 FCLK0 control register did not accept the requested divider")
    finally:
        slcr.close()


class PynqPlRuntime:
    """Direct PYNQ control path for the PS GP0/HP0 Budget-CUDA wrapper."""

    def __init__(self, bitfile: str | None = None, *, overlay=None, mmio=None, allocate_fn=None, register_base: int = PL_REGISTER_BASE):
        if overlay is None or mmio is None or allocate_fn is None:
            try:
                from pynq import MMIO, Overlay, allocate
            except ModuleNotFoundError as exc:
                raise RuntimeError("PynqPlRuntime must run on a PYNQ image or receive injected test backends") from exc
            overlay = overlay or Overlay(bitfile, download=True)
            mmio = mmio or MMIO(register_base, PL_REGISTER_RANGE)
            allocate_fn = allocate_fn or allocate
        self.overlay = overlay
        self.mmio = mmio
        self.allocate_fn = allocate_fn

    @classmethod
    def legacy_xlnk(cls, *, register_base: int = PL_REGISTER_BASE) -> "PynqPlRuntime":
        """Use direct MMIO and the legacy Xlnk CMA allocator on PYNQ-Z2."""
        return cls(overlay=object(), mmio=DirectMmio(register_base, PL_REGISTER_RANGE), allocate_fn=LegacyXlnkAllocator())

    def close(self) -> None:
        for resource in (self.mmio, self.allocate_fn):
            close = getattr(resource, "close", None)
            if close is not None:
                close()

    def allocate(self, shape, dtype: str | int | DType) -> PynqBuffer:
        dtype = normalize_dtype(dtype)
        numpy_dtype = {DType.INT8: "int8", DType.INT16: "int16", DType.INT32: "int32"}[dtype]
        data = self.allocate_fn(shape=tuple(shape), dtype=numpy_dtype)
        return PynqBuffer(data, dtype, int(data.nbytes))

    @staticmethod
    def _validate_buffer(buffer: PynqBuffer, minimum_bytes: int, alignment: int) -> None:
        if buffer.physical_address > 0xFFFFFFFF or buffer.physical_address % alignment or buffer.nbytes < minimum_bytes:
            raise ValueError("Buffer is not suitable for the 32-bit PL DDR path")

    def submit_gemm(self, a: PynqBuffer, b: PynqBuffer, c: PynqBuffer, m: int, n: int, k: int, *, array_count: int = 1) -> None:
        if a.dtype != DType.INT8 or b.dtype != DType.INT8 or c.dtype != DType.INT32:
            raise ValueError("PL GEMM requires INT8 A/B and INT32 C")
        if min(m, n, k) <= 0 or array_count != 1:
            raise ValueError("Invalid GEMM dimensions or array count")
        self._validate_buffer(a, m * k, 4)
        self._validate_buffer(b, k * n, 4)
        self._validate_buffer(c, m * n * 4, 4)
        a.flush()
        b.flush()
        self.mmio.write(PL_CONTROL, 0)
        self.mmio.write(PL_A_BASE, a.physical_address)
        self.mmio.write(PL_B_BASE, b.physical_address)
        self.mmio.write(PL_C_BASE, c.physical_address)
        self.mmio.write(PL_A_STRIDE, k)
        self.mmio.write(PL_B_STRIDE, n)
        self.mmio.write(PL_C_STRIDE, n * 4)
        self.mmio.write(PL_M, m)
        self.mmio.write(PL_N, n)
        self.mmio.write(PL_K, k)
        self.mmio.write(PL_TILE_SIZE, 16)
        self.mmio.write(PL_ARRAY_COUNT, array_count)
        self.mmio.write(PL_CONTROL, PL_START)

    def wait_gemm(self, output: PynqBuffer, timeout_polls: int = 1_000_000) -> None:
        for _ in range(timeout_polls):
            status = int(self.mmio.read(PL_STATUS))
            if status & PL_STATUS_ERROR:
                self.mmio.write(PL_CONTROL, 0)
                raise RuntimeError("Budget-CUDA PL reported an execution error")
            if status & PL_STATUS_DONE:
                self.mmio.write(PL_CONTROL, 0)
                output.invalidate()
                return
        self.mmio.write(PL_CONTROL, 0)
        raise TimeoutError("Budget-CUDA PL GEMM timed out")


class PynqDevice(Device):
    def __init__(self, address: str, port: int = 9000):
        super().__init__(address=address, port=port)

    def load_model(self, package: ModelPackage):
        return upload_model(self, package)


def connect(address: str, port: int = 9000) -> PynqDevice:
    return PynqDevice(address, port)
