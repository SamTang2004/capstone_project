import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda import Device
from budgetcuda.compiler import (
    BUFFER_DESC_STRUCT,
    LAYER_DESC_STRUCT,
    MODEL_DESC_STRUCT,
    MODEL_MAGIC,
    HostBuffer,
    Layer,
    ModelPackage,
    build_mlp_package,
    upload_model,
)
from budgetcuda.dtypes import BufferUsage, DType, OpType


class CompilerTest(unittest.TestCase):
    def test_host_buffer_from_values_packs_signed_int8_values(self):
        buffer = HostBuffer.from_values("w", (2, 2), "int8", [[1, -2], [3, -4]])
        self.assertEqual(buffer.size_bytes, 4)
        self.assertEqual(buffer.data, bytes([1, 254, 3, 252]))

    def test_layer_pack_matches_layer_desc_size(self):
        layer = Layer(OpType.GEMM, input0=1, output=2, weight=3, n=1, c=4, out_c=5)
        packed = layer.pack()
        self.assertEqual(len(packed), LAYER_DESC_STRUCT.size)
        self.assertEqual(int.from_bytes(packed[:4], "little"), OpType.GEMM)

    def test_model_package_serializes_header_table_and_layers(self):
        package = build_mlp_package(3, 2, [[1, 2], [3, 4], [5, 6]])
        descriptor = package.serialize_descriptor([10, 11, 12])
        self.assertEqual(int.from_bytes(descriptor[0:4], "little"), MODEL_MAGIC)
        self.assertEqual(len(descriptor), MODEL_DESC_STRUCT.size + 3 * BUFFER_DESC_STRUCT.size + LAYER_DESC_STRUCT.size)

    def test_build_mlp_package_uses_resident_weight_and_output_buffers(self):
        package = build_mlp_package(3, 2, [[1, 2], [3, 4], [5, 6]])
        self.assertEqual(len(package.buffers), 3)
        self.assertEqual(package.buffers[1].usage, BufferUsage.WEIGHT)
        self.assertEqual(package.buffers[2].dtype, DType.INT32)
        self.assertEqual(package.layers[0].op_type, OpType.GEMM)

    def test_upload_model_allocates_buffers_and_descriptor_on_device(self):
        dev = Device.mock()
        package = build_mlp_package(2, 2, [[1, 0], [0, 1]])
        model = upload_model(dev, package)
        self.assertEqual(model.input_tensor.shape, (1, 2))
        self.assertEqual(model.output_tensor.shape, (1, 2))
        self.assertEqual(len(model.weights), 1)
        self.assertEqual(model.descriptor_tensor.size_bytes, len(package.serialize_descriptor([1, 2, 3])))


if __name__ == "__main__":
    unittest.main()
