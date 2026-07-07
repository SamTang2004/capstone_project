import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda.quantization import dequantize_symmetric_int8, quantize_symmetric_int8


class QuantizationTest(unittest.TestCase):
    def test_quantize_symmetric_int8_computes_scale_when_missing(self):
        values = [-1.0, 0.0, 1.0]
        q, scale = quantize_symmetric_int8(values)
        self.assertEqual(scale, 1.0 / 127.0)
        self.assertEqual(q.tolist() if hasattr(q, "tolist") else q, [-127, 0, 127])

    def test_quantize_symmetric_int8_accepts_explicit_scale_and_clamps(self):
        q, scale = quantize_symmetric_int8([-200.0, 0.0, 200.0], scale=1.0)
        self.assertEqual(scale, 1.0)
        self.assertEqual(q.tolist() if hasattr(q, "tolist") else q, [-128, 0, 127])

    def test_dequantize_symmetric_int8_restores_float_values(self):
        out = dequantize_symmetric_int8([-2, 0, 2], scale=0.5)
        self.assertEqual(out.tolist() if hasattr(out, "tolist") else out, [-1.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
