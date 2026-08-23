from __future__ import annotations

from dataclasses import dataclass
import mmap
import os
import struct

from .compiler import ModelPackage, upload_model
from .device import Device
from .dtypes import DType, dtype_size, normalize_dtype
from .packing import pack_transformer_activations_int8, pack_transformer_weights


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
PL_RQ_MULTIPLIER = 0x34
PL_RQ_RIGHT_SHIFT = 0x38
PL_RQ_OUTPUT_ZERO_POINT = 0x3C
PL_DEBUG_CONTROL = 0x80
PL_TOTAL_CYCLES = 0x84
PL_BUSY_CYCLES = 0x8C
PL_MEMORY_STALL_CYCLES = 0x94
PL_INPUT_STALL_CYCLES = 0x9C
PL_OUTPUT_STALL_CYCLES = 0xA4
PL_LAST_OP_CYCLES = 0xAC
PL_MAX_OP_CYCLES = 0xB4
PL_COMPLETED_OPS = 0xBC
PL_DEBUG_STATUS = 0xC0
PL_TRACE_EVENT = 0xC4
PL_TRACE_DATA = 0xC8
PL_START = 0x1
PL_TRANSFORMER_WEIGHT_INT4 = 0x800
PL_TRANSFORMER_REQUANT_PACK = 0x1000
PL_STATUS_DONE = 0x2
PL_STATUS_ERROR = 0x4

XLNK_DEVICE = "/dev/xlnk"
XLNK_IOCALLOCBUF = 0xC0045802  # _IOWR('X', 2, unsigned long) on 32-bit ARM.
SLCR_BASE = 0xF8000000
SLCR_UNLOCK = 0x08
SLCR_LOCK = 0x04
FCLK0_CTRL = 0x170
FPGA_RST_CTRL = 0x240
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


class XrtBuffer:
    """CMA buffer backed by the board's ZOCL/XRT driver."""

    def __init__(self, device, nbytes: int):
        import pyxrt

        self._pyxrt = pyxrt
        self._bo = pyxrt.bo(device, int(nbytes), pyxrt.bo.flags.cacheable, 0)
        self.data = self._bo.map()
        self.nbytes = int(nbytes)
        self.physical_address = int(self._bo.address())

    def flush(self) -> None:
        self._bo.sync(self._pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, self.nbytes, 0)

    def invalidate(self) -> None:
        self._bo.sync(self._pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, self.nbytes, 0)

    def close(self) -> None:
        self.data = None
        self._bo = None


class XrtAllocator:
    """Allocator adapter for minimal PYNQ images that expose ZOCL but not PYNQ utilities."""

    _DTYPE_BYTES = {"int8": 1, "int16": 2, "int32": 4}

    def __init__(self, device_index: int = 0):
        import pyxrt

        self._device = pyxrt.device(device_index)

    def __call__(self, *, shape, dtype: str) -> XrtBuffer:
        try:
            item_bytes = self._DTYPE_BYTES[dtype]
        except KeyError as exc:
            raise ValueError(f"Unsupported XRT buffer dtype: {dtype}") from exc
        count = 1
        for dimension in shape:
            count *= int(dimension)
        if count <= 0:
            raise ValueError("XRT buffer shape must contain at least one element")
        return XrtBuffer(self._device, count * item_bytes)


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
        # Linux fpga_manager can leave FCLK0's fabric reset asserted after a
        # partial/full reconfiguration. Release only FCLK0 while preserving
        # the reset state of the other three fabric clocks.
        slcr.write(FPGA_RST_CTRL, slcr.read(FPGA_RST_CTRL) & ~0x1)
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

    @classmethod
    def xrt_direct(cls, *, register_base: int = PL_REGISTER_BASE, device_index: int = 0) -> "PynqPlRuntime":
        """Use direct MMIO with ZOCL/XRT CMA buffers on minimal board images."""
        return cls(overlay=object(), mmio=DirectMmio(register_base, PL_REGISTER_RANGE), allocate_fn=XrtAllocator(device_index))

    def close(self) -> None:
        for resource in (self.mmio, self.allocate_fn):
            close = getattr(resource, "close", None)
            if close is not None:
                close()

    def set_debug(self, *, enabled: bool, step: bool = False, freeze: bool = False, clear: bool = False) -> None:
        flags = int(enabled) | (int(step) << 1) | (int(freeze) << 2)
        if clear:
            # CLEAR is level-sensitive in RTL, so expose it as a host-side
            # pulse. Leaving it asserted would hold every counter at zero.
            self.mmio.write(PL_DEBUG_CONTROL, flags | 0x8)
        self.mmio.write(PL_DEBUG_CONTROL, flags)

    def _read_counter64(self, offset: int) -> int:
        while True:
            high1 = int(self.mmio.read(offset + 4))
            low = int(self.mmio.read(offset))
            high2 = int(self.mmio.read(offset + 4))
            if high1 == high2:
                return (high2 << 32) | low

    def debug_snapshot(self) -> dict[str, int | bool]:
        status = int(self.mmio.read(PL_DEBUG_STATUS))
        return {
            "total_cycles": self._read_counter64(PL_TOTAL_CYCLES),
            "busy_cycles": self._read_counter64(PL_BUSY_CYCLES),
            "memory_stall_cycles": self._read_counter64(PL_MEMORY_STALL_CYCLES),
            "input_stall_cycles": self._read_counter64(PL_INPUT_STALL_CYCLES),
            "output_stall_cycles": self._read_counter64(PL_OUTPUT_STALL_CYCLES),
            "last_op_cycles": self._read_counter64(PL_LAST_OP_CYCLES),
            "max_op_cycles": self._read_counter64(PL_MAX_OP_CYCLES),
            "completed_ops": int(self.mmio.read(PL_COMPLETED_OPS)),
            "active_op": status & 0xff,
            "error_latched": bool(status & 0x100),
            "trace_valid": bool(status & 0x200),
            "trace_event": int(self.mmio.read(PL_TRACE_EVENT)) & 0xff,
            "trace_data": int(self.mmio.read(PL_TRACE_DATA)),
        }

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

    def submit_transformer_gemm(self, packed_a: PynqBuffer, packed_b: PynqBuffer, c: PynqBuffer, m: int, n: int, k: int, *, weight_int4: bool = False) -> None:
        """Launch the 8x8 Transformer GEMM top using tile-major packed inputs."""
        if packed_a.dtype != DType.INT8 or packed_b.dtype != DType.INT8 or c.dtype != DType.INT32:
            raise ValueError("Transformer GEMM requires byte-packed A/B and INT32 C")
        if min(m, n, k) <= 0 or k > 1024:
            raise ValueError("Invalid Transformer GEMM dimensions")
        a_bytes = ((m + 7) // 8) * k * 8
        b_bytes = ((n + 7) // 8) * k * (4 if weight_int4 else 8)
        self._validate_buffer(packed_a, a_bytes, 4)
        self._validate_buffer(packed_b, b_bytes, 4)
        self._validate_buffer(c, m * n * 4, 4)
        packed_a.flush()
        packed_b.flush()
        self.mmio.write(PL_CONTROL, 0)
        self.mmio.write(PL_A_BASE, packed_a.physical_address)
        self.mmio.write(PL_B_BASE, packed_b.physical_address)
        self.mmio.write(PL_C_BASE, c.physical_address)
        self.mmio.write(PL_A_STRIDE, k * 8)
        self.mmio.write(PL_B_STRIDE, k * (4 if weight_int4 else 8))
        self.mmio.write(PL_C_STRIDE, n * 4)
        self.mmio.write(PL_M, m)
        self.mmio.write(PL_N, n)
        self.mmio.write(PL_K, k)
        self.mmio.write(PL_TILE_SIZE, 8)
        self.mmio.write(PL_ARRAY_COUNT, 1)
        self.mmio.write(PL_CONTROL, PL_START | (PL_TRANSFORMER_WEIGHT_INT4 if weight_int4 else 0))

    def submit_transformer_requantize_pack(
        self, source: PynqBuffer, packed_output: PynqBuffer, rows: int, columns: int,
        *, multiplier: int, right_shift: int, output_zero_point: int = 0,
    ) -> None:
        """Requantize row-major INT32 and emit tile-major INT8 activations."""
        if source.dtype != DType.INT32 or packed_output.dtype != DType.INT8:
            raise ValueError("Requantize-pack requires INT32 input and INT8 output")
        if min(rows, columns) <= 0 or rows > 0xffff or columns > 0xffff or not 0 <= right_shift <= 31:
            raise ValueError("Invalid requantize-pack dimensions or shift")
        output_bytes = ((rows + 7) // 8) * columns * 8
        self._validate_buffer(source, rows * columns * 4, 4)
        self._validate_buffer(packed_output, output_bytes, 4)
        source.flush()
        self.mmio.write(PL_CONTROL, 0)
        self.mmio.write(PL_A_BASE, source.physical_address)
        self.mmio.write(PL_C_BASE, packed_output.physical_address)
        self.mmio.write(PL_A_STRIDE, columns * 4)
        self.mmio.write(PL_M, rows)
        self.mmio.write(PL_N, columns)
        self.mmio.write(PL_RQ_MULTIPLIER, multiplier & 0xffffffff)
        self.mmio.write(PL_RQ_RIGHT_SHIFT, right_shift)
        self.mmio.write(PL_RQ_OUTPUT_ZERO_POINT, output_zero_point & 0xffffffff)
        self.mmio.write(PL_TILE_SIZE, 8)
        self.mmio.write(PL_ARRAY_COUNT, 1)
        self.mmio.write(PL_CONTROL, PL_START | PL_TRANSFORMER_REQUANT_PACK)

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
    def __init__(self, address: str | None = None, port: int | None = None, timeout: float | None = None):
        super().__init__(address=address, port=port, timeout=timeout)

    def load_model(self, package: ModelPackage):
        return upload_model(self, package)


def connect(address: str | None = None, port: int | None = None, timeout: float | None = None) -> PynqDevice:
    """Connect to a board runtime, using environment settings or the PYNQ default."""
    return PynqDevice(address, port, timeout)
