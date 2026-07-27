import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda.compiler import HostBuffer, ModelPackage
from budgetcuda.dtypes import BufferUsage, DType, Layout
from budgetcuda.onnx_import import OnnxImportError, _convert_qdq_package_to_int4


class Int4OnnxProfileTest(unittest.TestCase):
    def test_qdq_int8_containers_are_packed_without_changing_int32_buffers(self):
        package = ModelPackage(
            [
                HostBuffer("input", (1, 3), DType.INT8, Layout.NCHW, BufferUsage.ACTIVATION),
                HostBuffer.from_values("weight", (3,), DType.INT8, [-8, -1, 7]),
                HostBuffer("acc", (1, 1), DType.INT32),
            ],
            [],
            0,
            2,
        )
        converted = _convert_qdq_package_to_int4(package)
        self.assertEqual(converted.buffers[0].dtype, DType.INT4)
        self.assertEqual(converted.buffers[0].size_bytes, 2)
        self.assertEqual(converted.buffers[1].data, bytes([0xF8, 0x07]))
        self.assertEqual(converted.buffers[2].dtype, DType.INT32)

    def test_qdq_int4_profile_rejects_unconstrained_int8_weights(self):
        package = ModelPackage(
            [HostBuffer.from_values("weight", (2,), DType.INT8, [-9, 7])],
            [],
            0,
            0,
        )
        with self.assertRaisesRegex(OnnxImportError, "outside signed INT4"):
            _convert_qdq_package_to_int4(package)


if __name__ == "__main__":
    unittest.main()
