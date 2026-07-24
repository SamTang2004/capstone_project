import sys
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train_model"))


@unittest.skipUnless(torch is not None, "PyTorch is optional for host-only tests")
class BudgetCudaQatProfileTest(unittest.TestCase):
    def test_profile_prepares_and_keeps_signed_zero_centered_qparams(self):
        from qat_model import QATFruitCNN64, prepare_budgetcuda_qat

        model = prepare_budgetcuda_qat(QATFruitCNN64())
        model(torch.randn(2, 3, 64, 64))
        fake_quants = [module for module in model.modules() if isinstance(module, torch.ao.quantization.FakeQuantize)]
        self.assertGreater(len(fake_quants), 4)
        for module in fake_quants:
            self.assertEqual(module.dtype, torch.qint8)
            self.assertEqual(module.qscheme, torch.per_tensor_symmetric)
            self.assertTrue(torch.all(module.zero_point == 0))


if __name__ == "__main__":
    unittest.main()
