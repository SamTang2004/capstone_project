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
    GraphBuilder,
    Layer,
    ModelPackage,
    build_mlp_package,
    build_tiny_conv_relu_package,
    build_tiny_int4_cnn_package,
    upload_model,
)
from budgetcuda.dtypes import BufferUsage, DType, OpType


class CompilerTest(unittest.TestCase):
    def test_host_buffer_from_values_packs_signed_int8_values(self):
        buffer = HostBuffer.from_values("w", (2, 2), "int8", [[1, -2], [3, -4]])
        self.assertEqual(buffer.size_bytes, 4)
        self.assertEqual(buffer.data, bytes([1, 254, 3, 252]))

    def test_host_buffer_from_values_packs_signed_int4_nibbles(self):
        buffer = HostBuffer.from_values("w4", (1, 5), "int4", [-8, -1, 0, 7, 3])
        self.assertEqual(buffer.size_bytes, 3)
        self.assertEqual(buffer.data, bytes([0xF8, 0x70, 0x03]))

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
        self.assertEqual(package.buffers[0].dtype, DType.INT4)
        self.assertEqual(package.buffers[1].dtype, DType.INT4)
        self.assertEqual(len(package.buffers), 3)
        self.assertEqual(package.buffers[1].usage, BufferUsage.WEIGHT)
        self.assertEqual(package.buffers[2].dtype, DType.INT32)
        self.assertEqual(package.layers[0].op_type, OpType.GEMM)

    def test_graph_builder_creates_layer_sequence(self):
        graph = GraphBuilder("demo")
        inp = graph.add_buffer("input", (1, 1, 2, 2), DType.INT8, usage=BufferUsage.ACTIVATION)
        up = graph.add_buffer("up", (1, 1, 4, 4), DType.INT8, usage=BufferUsage.ACTIVATION)
        skip = graph.add_buffer("skip", (1, 1, 4, 4), DType.INT8, usage=BufferUsage.ACTIVATION)
        out = graph.add_buffer("out", (1, 2, 4, 4), DType.INT8, usage=BufferUsage.OUTPUT)
        graph.upsample(inp, up, n=1, c=1, h=2, w=2, out_h=4, out_w=4)
        graph.concat_channels(up, skip, out, n=1, h=4, w=4, out_c=2)
        package = graph.package(inp, out)
        self.assertEqual(package.input_index, inp)
        self.assertEqual(package.output_index, out)
        self.assertEqual([layer.op_type for layer in package.layers], [OpType.UPSAMPLE, OpType.CONCAT])

    def test_graph_builder_creates_requantize_layer_and_parameters(self):
        graph = GraphBuilder("quant")
        inp = graph.add_buffer("input", (4,), DType.INT32)
        params = graph.requantize_params("scale", multiplier=3, right_shift=1, output_zero_point=-2)
        out = graph.add_buffer("output", (4,), DType.INT8, usage=BufferUsage.OUTPUT)
        graph.requantize(inp, out, params)
        package = graph.package(inp, out)
        self.assertEqual(package.layers[0].op_type, OpType.REQUANTIZE)
        self.assertEqual(package.buffers[params].data, (3).to_bytes(4, "little", signed=True) + (1).to_bytes(4, "little") + bytes(4) + (-2).to_bytes(4, "little", signed=True))

    def test_build_tiny_conv_relu_package_uses_supported_ops(self):
        package = build_tiny_conv_relu_package((1, 1, 3, 3), [[[[1, 0], [0, 1]]]], bias_values=[-10])
        self.assertEqual(package.buffers[0].shape, (1, 1, 3, 3))
        self.assertEqual(package.buffers[0].dtype, DType.INT4)
        self.assertEqual(package.buffers[1].usage, BufferUsage.WEIGHT)
        self.assertEqual(package.buffers[package.output_index].dtype, DType.INT32)
        self.assertEqual([layer.op_type for layer in package.layers], [OpType.CONV2D_GEMM, OpType.ACTIVATION])

    def test_tiny_int4_cnn_runs_conv_requant_relu_end_to_end(self):
        package = build_tiny_int4_cnn_package(
            (1, 1, 3, 3),
            [1, -1, 2, 0],
            bias_values=[-1],
            multiplier=1,
            right_shift=0,
        )
        self.assertEqual(
            [layer.op_type for layer in package.layers],
            [OpType.CONV2D_GEMM, OpType.REQUANTIZE, OpType.ACTIVATION],
        )
        self.assertEqual(package.buffers[package.input_index].dtype, DType.INT4)
        self.assertEqual(package.buffers[package.output_index].dtype, DType.INT4)

        model = upload_model(Device.mock(), package)
        output = model.run([[[[1, 2, 3], [4, 5, 6], [7, 0, -1]]]])
        actual = output.reshape(-1).tolist() if hasattr(output, "reshape") else output
        self.assertEqual(actual, [6, 7, 7, 0])

    def test_upload_model_allocates_buffers_and_descriptor_on_device(self):
        dev = Device.mock()
        package = build_mlp_package(2, 2, [[1, 0], [0, 1]])
        model = upload_model(dev, package)
        self.assertEqual(model.input_tensor.shape, (1, 2))
        self.assertEqual(model.output_tensor.shape, (1, 2))
        self.assertEqual(len(model.weights), 1)
        self.assertEqual(len(model.tensors), len(package.buffers))
        self.assertEqual(model.descriptor_tensor.size_bytes, len(package.serialize_descriptor([1, 2, 3])))


if __name__ == "__main__":
    unittest.main()
