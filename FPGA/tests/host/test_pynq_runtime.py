import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda.dtypes import DType
from budgetcuda.pynq import PL_A_BASE, PL_ARRAY_COUNT, PL_CONTROL, PL_DEBUG_CONTROL, PL_STATUS, PL_STATUS_DONE, PL_TILE_SIZE, PL_TOTAL_CYCLES, PynqBuffer, PynqPlRuntime, pack_transformer_activations_int8, pack_transformer_weights


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
    def test_transformer_packers_use_tile_major_lane_order(self):
        activations = pack_transformer_activations_int8([[1, 2], [3, 4], [5, 6]], 3, 2)
        self.assertEqual(activations, bytes([1, 3, 5, 0, 0, 0, 0, 0, 2, 4, 6, 0, 0, 0, 0, 0]))
        weights8 = pack_transformer_weights([[1, -2], [3, -4]], 2, 2)
        self.assertEqual(weights8, bytes([1, 0xfe, 0, 0, 0, 0, 0, 0, 3, 0xfc, 0, 0, 0, 0, 0, 0]))
        weights4 = pack_transformer_weights([[1, -2], [3, -4]], 2, 2, int4=True)
        self.assertEqual(weights4, bytes([0xe1, 0, 0, 0, 0xc3, 0, 0, 0]))

    def test_submit_transformer_gemm_programs_w4_tile_contract(self):
        mmio = FakeMmio()
        runtime = PynqPlRuntime(overlay=object(), mmio=mmio, allocate_fn=lambda **_: None)
        a_raw, b_raw, c_raw = FakeBuffer(0x1000, 16), FakeBuffer(0x2000, 8), FakeBuffer(0x3000, 24)
        a = PynqBuffer(a_raw, DType.INT8, 16)
        b = PynqBuffer(b_raw, DType.INT8, 8)
        c = PynqBuffer(c_raw, DType.INT32, 24)
        runtime.submit_transformer_gemm(a, b, c, 3, 2, 2, weight_int4=True)
        self.assertIn((PL_TILE_SIZE, 8), mmio.writes)
        self.assertEqual(mmio.writes[-1], (PL_CONTROL, 0x801))
        self.assertEqual((a_raw.flush_count, b_raw.flush_count), (1, 1))

    def test_submit_transformer_requantize_pack_programs_layout_conversion(self):
        mmio = FakeMmio()
        runtime = PynqPlRuntime(overlay=object(), mmio=mmio, allocate_fn=lambda **_: None)
        source_raw, output_raw = FakeBuffer(0x1000, 36), FakeBuffer(0x2000, 24)
        source = PynqBuffer(source_raw, DType.INT32, 36)
        output = PynqBuffer(output_raw, DType.INT8, 24)
        runtime.submit_transformer_requantize_pack(source, output, 3, 3, multiplier=7, right_shift=2, output_zero_point=-1)
        self.assertIn((PL_TILE_SIZE, 8), mmio.writes)
        self.assertEqual(mmio.writes[-1], (PL_CONTROL, 0x1001))
        self.assertEqual(source_raw.flush_count, 1)

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

    def test_debug_control_and_atomic_cycle_snapshot(self):
        class CounterMmio(FakeMmio):
            def read(self, offset):
                self.last_read = offset
                return {PL_TOTAL_CYCLES: 5, PL_TOTAL_CYCLES + 4: 2}.get(offset, 0)
        mmio = CounterMmio()
        runtime = PynqPlRuntime(overlay=object(), mmio=mmio, allocate_fn=lambda **_: None)
        runtime.set_debug(enabled=True, step=True, freeze=True, clear=True)
        self.assertEqual(mmio.writes[-2:], [(PL_DEBUG_CONTROL, 15), (PL_DEBUG_CONTROL, 7)])
        snapshot = runtime.debug_snapshot()
        self.assertEqual(snapshot["total_cycles"], (2 << 32) | 5)
        self.assertFalse(snapshot["error_latched"])


if __name__ == "__main__":
    unittest.main()
