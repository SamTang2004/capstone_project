import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda.dtypes import DType
from budgetcuda.pynq import PL_A_BASE, PL_ARRAY_COUNT, PL_CONTROL, PL_STATUS, PL_STATUS_DONE, PynqBuffer, PynqPlRuntime


class FakeBuffer:
    def __init__(self, physical_address, nbytes):
        self.physical_address = physical_address
        self.nbytes = nbytes
        self.flush_count = 0
        self.invalidate_count = 0

    def flush(self):
        self.flush_count += 1

    def invalidate(self):
        self.invalidate_count += 1


class FakeMmio:
    def __init__(self):
        self.writes = []

    def write(self, offset, value):
        self.writes.append((offset, value))

    def read(self, offset):
        self.last_read = offset
        return PL_STATUS_DONE


class PynqRuntimeTest(unittest.TestCase):
    def test_submit_flushes_inputs_programs_physical_addresses_and_invalidates_output(self):
        mmio = FakeMmio()
        runtime = PynqPlRuntime(overlay=object(), mmio=mmio, allocate_fn=lambda **_: None)
        a_raw, b_raw, c_raw = FakeBuffer(0x1000, 256), FakeBuffer(0x2000, 256), FakeBuffer(0x3000, 1024)
        a, b, c = PynqBuffer(a_raw, DType.INT8, 256), PynqBuffer(b_raw, DType.INT8, 256), PynqBuffer(c_raw, DType.INT32, 1024)
        runtime.submit_gemm(a, b, c, 16, 16, 16)
        self.assertEqual(a_raw.flush_count, 1)
        self.assertEqual(b_raw.flush_count, 1)
        self.assertIn((PL_A_BASE, 0x1000), mmio.writes)
        self.assertIn((PL_ARRAY_COUNT, 1), mmio.writes)
        self.assertEqual(mmio.writes[-1], (PL_CONTROL, 1))
        runtime.wait_gemm(c)
        self.assertEqual(c_raw.invalidate_count, 1)
        self.assertEqual(mmio.last_read, PL_STATUS)

    def test_submit_rejects_address_outside_pl_address_width(self):
        runtime = PynqPlRuntime(overlay=object(), mmio=FakeMmio(), allocate_fn=lambda **_: None)
        a = PynqBuffer(FakeBuffer(0x1_0000_0000, 256), DType.INT8, 256)
        b = PynqBuffer(FakeBuffer(0x2000, 256), DType.INT8, 256)
        c = PynqBuffer(FakeBuffer(0x3000, 1024), DType.INT32, 1024)
        with self.assertRaises(ValueError):
            runtime.submit_gemm(a, b, c, 16, 16, 16)


if __name__ == "__main__":
    unittest.main()
