import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda.dtypes import DType, Layout, BufferUsage, dtype_size, normalize_dtype, normalize_layout, normalize_usage


class DTypesTest(unittest.TestCase):
    def test_normalize_dtype_accepts_names_and_enum_values(self):
        self.assertEqual(normalize_dtype("int8"), DType.INT8)
        self.assertEqual(normalize_dtype(DType.INT32), DType.INT32)
        self.assertEqual(normalize_dtype(2), DType.INT16)

    def test_normalize_dtype_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            normalize_dtype("float32")

    def test_normalize_layout_accepts_names_and_enum_values(self):
        self.assertEqual(normalize_layout("row_major"), Layout.ROW_MAJOR)
        self.assertEqual(normalize_layout(Layout.NCHW), Layout.NCHW)
        self.assertEqual(normalize_layout(3), Layout.NHWC)

    def test_normalize_usage_accepts_names_and_enum_values(self):
        self.assertEqual(normalize_usage("weight"), BufferUsage.WEIGHT)
        self.assertEqual(normalize_usage(BufferUsage.OUTPUT), BufferUsage.OUTPUT)
        self.assertEqual(normalize_usage(3), BufferUsage.WORKSPACE)

    def test_dtype_size_returns_bytes_per_element(self):
        self.assertEqual(dtype_size("int8"), 1)
        self.assertEqual(dtype_size("int16"), 2)
        self.assertEqual(dtype_size("int32"), 4)


if __name__ == "__main__":
    unittest.main()
