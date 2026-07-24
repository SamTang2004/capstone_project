import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

from budgetcuda.preprocessing import load_quantized_image
from budgetcuda.quantization import QuantizationParams


@unittest.skipUnless(np is not None and Image is not None, "numpy and Pillow are optional host dependencies")
class ImagePreprocessingTest(unittest.TestCase):
    def test_load_quantized_image_returns_batched_nchw_int8(self):
        pixels = np.array([[[0, 64, 127], [127, 64, 0]]], dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pixels.png"
            Image.fromarray(pixels, mode="RGB").save(path)
            result = load_quantized_image(
                path,
                (1, 3, 1, 2),
                QuantizationParams(1.0 / 255.0),
                layout="nchw",
            )

        self.assertEqual(result.shape, (1, 3, 1, 2))
        self.assertEqual(result.dtype, np.int8)
        self.assertEqual(result.tolist(), [[[[0, 127]], [[64, 64]], [[127, 0]]]])

    def test_load_quantized_image_validates_layout_and_normalization(self):
        with self.assertRaisesRegex(ValueError, "batch-one rank-4"):
            load_quantized_image("missing.png", (3, 4, 4), QuantizationParams(0.1))


if __name__ == "__main__":
    unittest.main()
