import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda.dtypes import DType
from budgetcuda.pynq import (
    PL_ARRAY_COUNT,
    PL_A_STRIDE,
    PL_B_STRIDE,
    PL_CONTROL,
    PL_CONTROL_INT4,
    PL_START,
    PL_TILE_SIZE,
    PynqPlRuntime,
)


class FakeAllocation:
    def __init__(self, shape, dtype, address):
        item_size = {"uint8": 1, "int8": 1, "int16": 2, "int32": 4}[dtype]
        count = 1
        for dimension in shape:
            count *= dimension
        self.nbytes = count * item_size
        self.physical_address = address
        self.flush_count = 0

    def flush(self):
        self.flush_count += 1

    def invalidate(self):
        pass


class FakeAllocator:
    def __init__(self):
        self.calls = []

    def __call__(self, *, shape, dtype):
        self.calls.append((shape, dtype))
        return FakeAllocation(shape, dtype, 0x1000 + 0x1000 * len(self.calls))


class FakeMmio:
    def __init__(self):
        self.registers = {}

    def write(self, offset, value):
        self.registers[offset] = value

    def read(self, offset):
        return self.registers.get(offset, 0)


class PynqInt4RuntimeTest(unittest.TestCase):
    def test_allocate_uses_uint8_storage_for_packed_int4(self):
        allocator = FakeAllocator()
        runtime = PynqPlRuntime(overlay=object(), mmio=FakeMmio(), allocate_fn=allocator)
        buffer = runtime.allocate((3, 5), DType.INT4)
        self.assertEqual(allocator.calls, [((8,), "uint8")])
        self.assertEqual(buffer.shape, (3, 5))
        self.assertEqual(buffer.nbytes, 8)

    def test_submit_gemm_programs_dual_int4_path(self):
        allocator = FakeAllocator()
        mmio = FakeMmio()
        runtime = PynqPlRuntime(overlay=object(), mmio=mmio, allocate_fn=allocator)
        a = runtime.allocate((8, 8), "int4")
        b = runtime.allocate((8, 16), "int4")
        c = runtime.allocate((8, 16), "int32")
        runtime.submit_gemm(a, b, c, 8, 16, 8)
        self.assertEqual(mmio.registers[PL_A_STRIDE], 4)
        self.assertEqual(mmio.registers[PL_B_STRIDE], 8)
        self.assertEqual(mmio.registers[PL_TILE_SIZE], 8)
        self.assertEqual(mmio.registers[PL_ARRAY_COUNT], 2)
        self.assertEqual(mmio.registers[PL_CONTROL], PL_START | PL_CONTROL_INT4)
        self.assertEqual((a.data.flush_count, b.data.flush_count), (1, 1))


if __name__ == "__main__":
    unittest.main()
