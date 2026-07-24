import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper
except ModuleNotFoundError:
    onnx = None

from budgetcuda import Device
from budgetcuda.dtypes import DType, OpType
from budgetcuda.onnx_import import OnnxImportError, import_onnx_model


@unittest.skipUnless(onnx is not None, "onnx and numpy are optional host dependencies")
class OnnxImportTest(unittest.TestCase):
    def test_qlinear_conv_relu_lowers_to_conv_requantize_activation_and_runs(self):
        inputs = [helper.make_tensor_value_info("input", TensorProto.INT8, [1, 1, 3, 3])]
        outputs = [helper.make_tensor_value_info("output", TensorProto.INT8, [1, 1, 2, 2])]
        initializers = [
            helper.make_tensor("weight", TensorProto.INT8, [1, 1, 2, 2], [1, 0, 0, 1]),
            helper.make_tensor("x_scale", TensorProto.FLOAT, [], [0.5]),
            helper.make_tensor("w_scale", TensorProto.FLOAT, [], [0.25]),
            helper.make_tensor("y_scale", TensorProto.FLOAT, [], [0.125]),
            helper.make_tensor("x_zp", TensorProto.INT8, [], [0]),
            helper.make_tensor("w_zp", TensorProto.INT8, [], [0]),
            helper.make_tensor("y_zp", TensorProto.INT8, [], [0]),
            helper.make_tensor("bias", TensorProto.INT32, [1], [-10]),
        ]
        qconv = helper.make_node("QLinearConv", ["input", "x_scale", "x_zp", "weight", "w_scale", "w_zp", "y_scale", "y_zp", "bias"], ["conv"], kernel_shape=[2, 2])
        relu = helper.make_node("Relu", ["conv"], ["output"])
        model = helper.make_model(helper.make_graph([qconv, relu], "qconv", inputs, outputs, initializers), opset_imports=[helper.make_opsetid("", 13)])

        package = import_onnx_model(model, name="tiny")
        self.assertEqual([layer.op_type for layer in package.layers], [OpType.CONV2D_GEMM, OpType.REQUANTIZE, OpType.ACTIVATION])
        self.assertEqual(package.buffers[package.output_index].dtype, DType.INT8)
        result = Device.mock().load_model(package).run([-4, 2, 3, 8, 5, 6, 7, 8, 9])
        values = result.reshape(-1).tolist() if hasattr(result, "reshape") else result
        self.assertEqual(values, [0, 0, 6, 4])

    def test_rejects_nonzero_qlinearconv_input_zero_point(self):
        inputs = [helper.make_tensor_value_info("input", TensorProto.INT8, [1, 1, 2, 2])]
        outputs = [helper.make_tensor_value_info("output", TensorProto.INT8, [1, 1, 1, 1])]
        initializers = [
            helper.make_tensor("weight", TensorProto.INT8, [1, 1, 2, 2], [1, 1, 1, 1]),
            helper.make_tensor("scale", TensorProto.FLOAT, [], [1.0]),
            helper.make_tensor("x_zp", TensorProto.INT8, [], [1]),
            helper.make_tensor("zp", TensorProto.INT8, [], [0]),
        ]
        node = helper.make_node("QLinearConv", ["input", "scale", "x_zp", "weight", "scale", "zp", "scale", "zp"], ["output"], kernel_shape=[2, 2])
        model = helper.make_model(helper.make_graph([node], "invalid", inputs, outputs, initializers), opset_imports=[helper.make_opsetid("", 13)])
        with self.assertRaisesRegex(OnnxImportError, "zero point"):
            import_onnx_model(model)

    def test_averagepool_and_globalaveragepool_lower_to_average_pool_layers(self):
        inputs = [helper.make_tensor_value_info("input", TensorProto.INT8, [1, 1, 3, 3])]
        outputs = [helper.make_tensor_value_info("output", TensorProto.INT8, [1, 1, 1, 1])]
        avg = helper.make_node("AveragePool", ["input"], ["pooled"], kernel_shape=[2, 2], strides=[1, 1])
        global_avg = helper.make_node("GlobalAveragePool", ["pooled"], ["output"])
        model = helper.make_model(helper.make_graph([avg, global_avg], "pools", inputs, outputs), opset_imports=[helper.make_opsetid("", 13)])
        package = import_onnx_model(model, name="pools")
        self.assertEqual([layer.op_type for layer in package.layers], [OpType.POOL, OpType.POOL])
        self.assertEqual([layer.flags for layer in package.layers], [1, 1])
        self.assertEqual((package.layers[1].kernel_h, package.layers[1].kernel_w), (2, 2))
        self.assertEqual(Device.mock().load_model(package).run([-7, -2, 3, 8, 4, -5, 6, 9, -1]).reshape(-1).tolist(), [1])

    def test_signed_per_tensor_qdq_conv_and_averagepool_lower(self):
        x = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 1, 3, 3])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 1, 1])
        def tensor(name, dtype, dims, values):
            return helper.make_tensor(name, dtype, dims, values)
        initializers = [
            tensor("xs", TensorProto.FLOAT, [], [0.25]), tensor("xz", TensorProto.INT8, [], [0]),
            tensor("w", TensorProto.INT8, [1, 1, 2, 2], [1, 0, 0, 1]), tensor("ws", TensorProto.FLOAT, [], [0.5]), tensor("wz", TensorProto.INT8, [], [0]),
            tensor("b", TensorProto.INT32, [1], [4]), tensor("bs", TensorProto.FLOAT, [], [0.125]), tensor("bz", TensorProto.INT32, [], [0]),
            tensor("ys", TensorProto.FLOAT, [], [0.125]), tensor("yz", TensorProto.INT8, [], [0]),
        ]
        nodes = [
            helper.make_node("DequantizeLinear", ["x", "xs", "xz"], ["xd"]),
            helper.make_node("DequantizeLinear", ["w", "ws", "wz"], ["wd"]),
            helper.make_node("DequantizeLinear", ["b", "bs", "bz"], ["bd"]),
            helper.make_node("Conv", ["xd", "wd", "bd"], ["conv"], kernel_shape=[2, 2]),
            helper.make_node("QuantizeLinear", ["conv", "ys", "yz"], ["qconv"]),
            helper.make_node("DequantizeLinear", ["qconv", "ys", "yz"], ["poolin"]),
            helper.make_node("AveragePool", ["poolin"], ["pool"], kernel_shape=[2, 2]),
            helper.make_node("QuantizeLinear", ["pool", "ys", "yz"], ["qpool"]),
            helper.make_node("DequantizeLinear", ["qpool", "ys", "yz"], ["y"]),
        ]
        model = helper.make_model(helper.make_graph(nodes, "signed_qdq", [x], [y], initializers), opset_imports=[helper.make_opsetid("", 13)])
        package = import_onnx_model(model)
        self.assertEqual([layer.op_type for layer in package.layers], [OpType.CONV2D_GEMM, OpType.REQUANTIZE, OpType.POOL])
        self.assertEqual(package.layers[-1].flags, 1)

    def test_signed_per_tensor_qdq_flatten_and_gemm_lower(self):
        x = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 1, 2, 2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
        def tensor(name, dtype, dims, values):
            return helper.make_tensor(name, dtype, dims, values)
        initializers = [
            tensor("s", TensorProto.FLOAT, [], [0.25]), tensor("z", TensorProto.INT8, [], [0]),
            tensor("w", TensorProto.INT8, [4, 2], [1, 0, 0, 1, 1, 1, -1, 1]), tensor("ws", TensorProto.FLOAT, [], [0.5]), tensor("wz", TensorProto.INT8, [], [0]),
            tensor("ys", TensorProto.FLOAT, [], [0.125]), tensor("yz", TensorProto.INT8, [], [0]),
        ]
        nodes = [
            helper.make_node("DequantizeLinear", ["x", "s", "z"], ["xd"]),
            helper.make_node("Flatten", ["xd"], ["flat"]), helper.make_node("QuantizeLinear", ["flat", "s", "z"], ["qflat"]),
            helper.make_node("DequantizeLinear", ["qflat", "s", "z"], ["gemmin"]), helper.make_node("DequantizeLinear", ["w", "ws", "wz"], ["wd"]),
            helper.make_node("Gemm", ["gemmin", "wd"], ["gemm"]), helper.make_node("QuantizeLinear", ["gemm", "ys", "yz"], ["qgemm"]),
            helper.make_node("DequantizeLinear", ["qgemm", "ys", "yz"], ["y"]),
        ]
        model = helper.make_model(helper.make_graph(nodes, "qdq_gemm", [x], [y], initializers), opset_imports=[helper.make_opsetid("", 13)])
        package = import_onnx_model(model)
        self.assertEqual([layer.op_type for layer in package.layers], [OpType.RESHAPE, OpType.GEMM, OpType.REQUANTIZE])

    def test_signed_qdq_biased_transposed_gemm_rewrites_to_1x1_conv(self):
        x = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
        def tensor(name, dtype, dims, values):
            return helper.make_tensor(name, dtype, dims, values)
        initializers = [
            tensor("s", TensorProto.FLOAT, [], [0.25]), tensor("z", TensorProto.INT8, [], [0]),
            tensor("w", TensorProto.INT8, [2, 4], [1, 0, 0, 1, 1, 1, -1, 1]), tensor("ws", TensorProto.FLOAT, [], [0.5]), tensor("wz", TensorProto.INT8, [], [0]),
            tensor("b", TensorProto.INT32, [2], [4, -4]), tensor("bs", TensorProto.FLOAT, [], [0.125]), tensor("bz", TensorProto.INT32, [], [0]),
            tensor("ys", TensorProto.FLOAT, [], [0.125]), tensor("yz", TensorProto.INT8, [], [0]),
        ]
        nodes = [
            helper.make_node("DequantizeLinear", ["x", "s", "z"], ["xd"]), helper.make_node("DequantizeLinear", ["w", "ws", "wz"], ["wd"]), helper.make_node("DequantizeLinear", ["b", "bs", "bz"], ["bd"]),
            helper.make_node("Gemm", ["xd", "wd", "bd"], ["gemm"], transB=1), helper.make_node("QuantizeLinear", ["gemm", "ys", "yz"], ["qgemm"]),
            helper.make_node("DequantizeLinear", ["qgemm", "ys", "yz"], ["y"]),
        ]
        model = helper.make_model(helper.make_graph(nodes, "qdq_classifier", [x], [y], initializers), opset_imports=[helper.make_opsetid("", 13)])
        package = import_onnx_model(model)
        self.assertEqual([layer.op_type for layer in package.layers], [OpType.RESHAPE, OpType.CONV2D_GEMM, OpType.REQUANTIZE, OpType.RESHAPE])
        self.assertNotEqual(package.layers[1].bias, 0)
        result = Device.mock().load_model(package).run([1, 2, 3, 4])
        self.assertEqual(result.reshape(-1).tolist(), [9, 0])

    def test_signed_qdq_cnn_matches_onnx_runtime_end_to_end(self):
        try:
            import onnxruntime as ort
        except ModuleNotFoundError:
            self.skipTest("onnxruntime is optional for end-to-end comparison")

        x = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 1, 4, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])

        def tensor(name, dtype, dims, values):
            return helper.make_tensor(name, dtype, dims, values)

        initializers = [
            tensor("xs", TensorProto.FLOAT, [], [0.25]), tensor("xz", TensorProto.INT8, [], [0]),
            tensor("cw", TensorProto.INT8, [2, 1, 1, 1], [1, -1]), tensor("cws", TensorProto.FLOAT, [], [0.5]), tensor("cwz", TensorProto.INT8, [], [0]),
            tensor("cb", TensorProto.INT32, [2], [0, 20]), tensor("cbs", TensorProto.FLOAT, [], [0.125]), tensor("cbz", TensorProto.INT32, [], [0]),
            tensor("cs", TensorProto.FLOAT, [], [0.125]), tensor("cz", TensorProto.INT8, [], [0]),
            tensor("fw", TensorProto.INT8, [3, 2], [1, 1, 1, -1, -1, 1]), tensor("fws", TensorProto.FLOAT, [], [0.5]), tensor("fwz", TensorProto.INT8, [], [0]),
            tensor("fb", TensorProto.INT32, [3], [0, 0, 0]), tensor("fbs", TensorProto.FLOAT, [], [0.0625]), tensor("fbz", TensorProto.INT32, [], [0]),
            tensor("ys", TensorProto.FLOAT, [], [0.125]), tensor("yz", TensorProto.INT8, [], [0]),
        ]
        nodes = [
            helper.make_node("DequantizeLinear", ["x", "xs", "xz"], ["xd"]),
            helper.make_node("DequantizeLinear", ["cw", "cws", "cwz"], ["cwd"]),
            helper.make_node("DequantizeLinear", ["cb", "cbs", "cbz"], ["cbd"]),
            helper.make_node("Conv", ["xd", "cwd", "cbd"], ["conv"], kernel_shape=[1, 1]),
            helper.make_node("QuantizeLinear", ["conv", "cs", "cz"], ["qconv"]),
            helper.make_node("DequantizeLinear", ["qconv", "cs", "cz"], ["relud"]),
            helper.make_node("Relu", ["relud"], ["relu"]),
            helper.make_node("QuantizeLinear", ["relu", "cs", "cz"], ["qrelu"]),
            helper.make_node("DequantizeLinear", ["qrelu", "cs", "cz"], ["poold"]),
            helper.make_node("GlobalAveragePool", ["poold"], ["pool"]),
            helper.make_node("QuantizeLinear", ["pool", "cs", "cz"], ["qpool"]),
            helper.make_node("DequantizeLinear", ["qpool", "cs", "cz"], ["flatd"]),
            helper.make_node("Flatten", ["flatd"], ["flat"]),
            helper.make_node("QuantizeLinear", ["flat", "cs", "cz"], ["qflat"]),
            helper.make_node("DequantizeLinear", ["qflat", "cs", "cz"], ["gemmd"]),
            helper.make_node("DequantizeLinear", ["fw", "fws", "fwz"], ["fwd"]),
            helper.make_node("DequantizeLinear", ["fb", "fbs", "fbz"], ["fbd"]),
            helper.make_node("Gemm", ["gemmd", "fwd", "fbd"], ["gemm"], transB=1),
            helper.make_node("QuantizeLinear", ["gemm", "ys", "yz"], ["qlogits"]),
            helper.make_node("DequantizeLinear", ["qlogits", "ys", "yz"], ["y"]),
        ]
        model = helper.make_model(helper.make_graph(nodes, "qdq_cnn", [x], [y], initializers), opset_imports=[helper.make_opsetid("", 13)])
        onnx.checker.check_model(model)
        runtime_input = np.array([[[[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4], [5, 6, 7, 8]]]], dtype=np.int8)
        reference = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": runtime_input})[0]

        package = import_onnx_model(model, name="qdq_cnn")
        result = Device.mock().load_model(package).run(runtime_input.reshape(-1).tolist())
        actual = result.reshape(1, 3).astype(np.float32) * np.float32(0.125)

        # The deployed AveragePool divider truncates toward zero to keep the
        # PL implementation compact. ONNX QuantizeLinear rounds instead; this
        # graph's classifier scales make that bounded difference one output LSB.
        np.testing.assert_allclose(actual, reference, rtol=0, atol=0.125)
        self.assertEqual(actual.reshape(-1).tolist(), [1.125, -0.75, 0.625])


class OnnxImportDependencyTest(unittest.TestCase):
    def test_missing_onnx_has_actionable_error(self):
        if onnx is not None:
            self.skipTest("ONNX is installed")
        with self.assertRaisesRegex(OnnxImportError, "pip install onnx"):
            import_onnx_model("model.onnx")


if __name__ == "__main__":
    unittest.main()
