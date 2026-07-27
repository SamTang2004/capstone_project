import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_MODEL = ROOT / "train_model"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRAIN_MODEL))

try:
    import onnx
    import onnxruntime as ort
    import torch
except ModuleNotFoundError:
    onnx = None
    ort = None
    torch = None

from budgetcuda.dtypes import DType
from budgetcuda.onnx_import import import_onnx_model


@unittest.skipUnless(torch is not None and onnx is not None, "torch and onnx are optional")
class Int4TrainingExportTest(unittest.TestCase):
    def test_qat_profile_exports_and_reimports_as_packed_int4(self):
        from export_budgetcuda_int4_onnx import export_int4_qdq
        from qat_model import QATFruitCNN64, prepare_budgetcuda_qat

        model = prepare_budgetcuda_qat(QATFruitCNN64())
        model.train()
        with torch.no_grad():
            model(torch.rand(2, 3, 64, 64))
        for name, module in model.named_modules():
            if name.endswith("weight_fake_quant") or name.endswith("activation_post_process"):
                if hasattr(module, "quant_min"):
                    self.assertEqual((module.quant_min, module.quant_max), (-8, 7))
        self.assertEqual(type(model.quant.activation_post_process).__name__, "TruncatingFakeQuantize")
        self.assertIs(model.avgpool._scale_source, model.conv3.activation_post_process)

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "fruit_int4.onnx"
            export_int4_qdq(model, model_path)
            package = import_onnx_model(model_path)
            if ort is not None:
                session = ort.InferenceSession(
                    str(model_path), providers=["CPUExecutionProvider"]
                )
                result = session.run(
                    None,
                    {
                        session.get_inputs()[0].name: torch.zeros(
                            1, 3, 64, 64, dtype=torch.int8
                        ).numpy()
                    },
                )[0]
                self.assertEqual(result.shape, (1, 3))
                exported = onnx.load(model_path)
                clips = [node for node in exported.graph.node if node.op_type == "Clip"]
                self.assertGreater(len(clips), 0)

        self.assertEqual(package.buffers[package.input_index].dtype, DType.INT4)
        self.assertEqual(package.buffers[package.output_index].dtype, DType.INT4)
        self.assertGreater(len(package.layers), 10)
        self.assertTrue(all(buffer.dtype != DType.INT8 for buffer in package.buffers))


if __name__ == "__main__":
    unittest.main()
