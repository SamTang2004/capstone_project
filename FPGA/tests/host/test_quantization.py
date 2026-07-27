import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    import onnx
    from onnx import TensorProto, helper
except ModuleNotFoundError:
    onnx = None

from budgetcuda.quantization import (
    Int4QuantizationParams,
    QuantizationParams,
    dequantize_int4,
    dequantize_int8,
    dequantize_symmetric_int8,
    onnx_qdq_params,
    quantize_int8,
    quantize_int4,
    quantize_symmetric_int8,
)


class QuantizationTest(unittest.TestCase):
    def test_affine_int4_quantization_clamps_and_preserves_shape(self):
        values = [[-10.0, -1.0], [0.0, 10.0]]
        quantized = quantize_int4(values, scale=1.0, zero_point=0)
        actual = quantized.tolist() if hasattr(quantized, "tolist") else quantized
        self.assertEqual(actual, [[-8, -1], [0, 7]] if np is not None else [-8, -1, 0, 7])
        restored = dequantize_int4(quantized, scale=0.5, zero_point=0)
        actual = restored.tolist() if hasattr(restored, "tolist") else restored
        self.assertEqual(actual, [[-4.0, -0.5], [0.0, 3.5]] if np is not None else [-4.0, -0.5, 0.0, 3.5])

    def test_int4_params_validate_zero_point(self):
        with self.assertRaisesRegex(ValueError, "-8..7"):
            Int4QuantizationParams(0.25, 8)

    def test_int4_quantization_truncates_toward_zero(self):
        quantized = quantize_int4([-1.9, -0.9, 0.9, 1.9], scale=1.0)
        actual = quantized.tolist() if hasattr(quantized, "tolist") else quantized
        self.assertEqual(actual, [-1, 0, 0, 1])

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

    def test_affine_helpers_preserve_numpy_shape_and_apply_zero_point(self):
        if np is None:
            self.skipTest("numpy is an optional host dependency")
        values = np.array([[-1.0, 0.0], [1.0, 100.0]], dtype=np.float32)
        quantized = quantize_int8(values, scale=0.5, zero_point=3)
        self.assertEqual(quantized.shape, values.shape)
        self.assertEqual(quantized.dtype, np.int8)
        self.assertEqual(quantized.tolist(), [[1, 3], [5, 127]])
        np.testing.assert_array_equal(
            dequantize_int8(quantized[:, :1], scale=0.5, zero_point=3),
            np.array([[-1.0], [1.0]], dtype=np.float32),
        )

    def test_quantization_params_validate_and_delegate(self):
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            QuantizationParams(0.0)
        with self.assertRaisesRegex(ValueError, "-128..127"):
            QuantizationParams(0.25, 128)
        params = QuantizationParams(0.25, -2)
        quantized = params.quantize([0.0])
        dequantized = params.dequantize([-2])
        self.assertEqual(quantized.tolist() if hasattr(quantized, "tolist") else quantized, [-2])
        self.assertEqual(dequantized.tolist() if hasattr(dequantized, "tolist") else dequantized, [0.0])

    @unittest.skipUnless(onnx is not None, "onnx and numpy are optional host dependencies")
    def test_onnx_qdq_params_extract_input_and_output_metadata(self):
        initializers = [
            helper.make_tensor("input_scale", TensorProto.FLOAT, [], [0.25]),
            helper.make_tensor("input_zp", TensorProto.INT8, [], [-3]),
            helper.make_tensor("output_scale", TensorProto.FLOAT, [], [0.5]),
            helper.make_tensor("output_zp", TensorProto.INT8, [], [4]),
        ]
        nodes = [
            helper.make_node("DequantizeLinear", ["input", "input_scale", "input_zp"], ["values"]),
            helper.make_node("QuantizeLinear", ["values", "output_scale", "output_zp"], ["quantized"]),
            helper.make_node("DequantizeLinear", ["quantized", "output_scale", "output_zp"], ["output"]),
        ]
        model = helper.make_model(
            helper.make_graph(
                nodes,
                "qparams",
                [helper.make_tensor_value_info("input", TensorProto.INT8, [1, 2])],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
                initializers,
            ),
            opset_imports=[helper.make_opsetid("", 13)],
        )

        params = onnx_qdq_params(model)
        self.assertEqual((params.input_name, params.output_name), ("input", "output"))
        self.assertEqual(params.input, Int4QuantizationParams(0.25, -3))
        self.assertEqual(params.output, Int4QuantizationParams(0.5, 4))

        legacy = onnx_qdq_params(model, target_dtype="int8")
        self.assertEqual(legacy.input, QuantizationParams(0.25, -3))
        self.assertEqual(legacy.output, QuantizationParams(0.5, 4))


if __name__ == "__main__":
    unittest.main()
