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
    CONV_FLAG_PRESLICED_WEIGHTS,
    HostBuffer,
    GraphBuilder,
    Layer,
    ModelPackage,
    build_mlp_package,
    build_tiny_conv_relu_package,
    preslice_conv2d_weights_int8,
    upload_model,
)
from budgetcuda.dtypes import BufferUsage, DType, Layout, OpType


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
        self.assertEqual(package.buffers[1].usage, BufferUsage.WEIGHT)
        self.assertEqual(package.buffers[package.output_index].dtype, DType.INT32)
        self.assertEqual([layer.op_type for layer in package.layers], [OpType.CONV2D_GEMM, OpType.ACTIVATION])

    def test_conv_weights_are_presliced_into_array_consumption_order(self):
        raw = bytes(range(18))  # O=2, I=1, KH=3, KW=3
        packed = preslice_conv2d_weights_int8(
            raw, out_channels=2, input_channels=1, kernel_h=3, kernel_w=3
        )
        self.assertEqual(len(packed), 256)
        self.assertEqual(list(packed[0:2]), [0, 9])
        self.assertEqual(list(packed[16:18]), [1, 10])
        self.assertEqual(list(packed[8 * 16 : 8 * 16 + 2]), [8, 17])
        self.assertEqual(set(packed[9 * 16 :]), {0})

    def test_graph_builder_marks_and_physically_sizes_presliced_conv_weights(self):
        graph = GraphBuilder("presliced")
        inp = graph.add_buffer("input", (1, 1, 4, 4), DType.INT8, layout=Layout.NCHW)
        weight = graph.add_buffer("weight", (2, 1, 3, 3), DType.INT8, BufferUsage.WEIGHT, data=list(range(18)), layout=Layout.NCHW)
        out = graph.add_buffer("output", (1, 2, 2, 2), DType.INT32, BufferUsage.OUTPUT, layout=Layout.NCHW)
        graph.conv2d(inp, weight, out, n=1, c=1, h=4, w=4, out_c=2, out_h=2, out_w=2, kernel_h=3, kernel_w=3)
        self.assertEqual(graph.buffers[weight].layout, Layout.CONV_WEIGHT_TILED_INT8)
        self.assertEqual(graph.buffers[weight].shape, (1, 1, 16, 16))
        self.assertEqual(graph.layers[0].flags, CONV_FLAG_PRESLICED_WEIGHTS)

    def test_presliced_conv_matches_raw_conv_in_mock_backend(self):
        input_values = bytes(range(1, 17))
        weights = [[[[1, -1], [2, 0]]], [[[0, 2], [-1, 1]]]]
        expected = None
        for preslice in (False, True):
            graph = GraphBuilder(f"compare_{preslice}")
            inp = graph.add_buffer("input", (1, 1, 4, 4), DType.INT8, BufferUsage.ACTIVATION, layout=Layout.NCHW)
            weight = graph.add_buffer("weight", (2, 1, 2, 2), DType.INT8, BufferUsage.WEIGHT, data=weights, layout=Layout.NCHW)
            out = graph.add_buffer("output", (1, 2, 3, 3), DType.INT32, BufferUsage.OUTPUT, layout=Layout.NCHW)
            graph.conv2d(inp, weight, out, n=1, c=1, h=4, w=4, out_c=2, out_h=3, out_w=3, kernel_h=2, kernel_w=2, preslice_weights=preslice)
            model = upload_model(Device.mock(), graph.package(inp, out))
            model.run(input_values)
            result = bytes(model.device.read(model.output_tensor))
            if expected is None:
                expected = result
            else:
                self.assertEqual(result, expected)

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
