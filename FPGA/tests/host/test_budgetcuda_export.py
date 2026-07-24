import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train_model"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import onnx
    import torch
except ModuleNotFoundError:
    onnx = None

from budgetcuda.dtypes import OpType
from budgetcuda import Device
from budgetcuda.onnx_import import import_onnx_model
from export_budgetcuda_int8_onnx import DEFAULT_CHECKPOINT, export_model, load_model


@unittest.skipUnless(onnx is not None and DEFAULT_CHECKPOINT.exists(), "requires ONNX and the signed QAT smoke checkpoint")
class BudgetCudaExportTest(unittest.TestCase):
    def test_signed_qat_export_is_importable_and_covers_the_fruit_cnn(self):
        model, _ = load_model(DEFAULT_CHECKPOINT)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fruit.onnx"
            export_model(model, path)
            graph = onnx.load(path).graph
            self.assertEqual(graph.input[0].type.tensor_type.elem_type, onnx.TensorProto.INT8)
            package = import_onnx_model(path)
        ops = [layer.op_type for layer in package.layers]
        self.assertEqual(ops.count(OpType.CONV2D_GEMM), 4)
        self.assertEqual(ops.count(OpType.REQUANTIZE), 4)
        self.assertEqual(ops.count(OpType.ACTIVATION), 3)
        self.assertEqual(ops.count(OpType.POOL), 3)
        self.assertIn(OpType.RESHAPE, ops)

    def test_exported_model_runs_through_the_virtual_pl_path(self):
        model, _ = load_model(DEFAULT_CHECKPOINT)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "fruit.onnx"
            export_model(model, path)
            package = import_onnx_model(path)
        # The zero-valued signed input is deterministic and still traverses
        # every convolution, pool, requantizer, and classifier stage.
        result = Device.mock().load_model(package).run([0] * (3 * 64 * 64))
        self.assertEqual(result.shape, (1, 3))
        self.assertTrue(((result >= -128) & (result <= 127)).all())


if __name__ == "__main__":
    unittest.main()
