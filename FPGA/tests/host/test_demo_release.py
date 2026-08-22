from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import unittest


FPGA_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = FPGA_ROOT.parent
DEMO_ROOT = FPGA_ROOT / "demo"
DATASET_ROOT = DEMO_ROOT / "dataset" / "fruits-360-100x100"
MODEL = DEMO_ROOT / "models" / "fruit3_budgetcuda_int8.onnx"
MODEL_SHA256 = "a36f9701f126c6a76fb81366d4a384d9b920469749e5e32d2604e47e33a9b348"
REQUIREMENTS = REPO_ROOT / "requirements.txt"
REQUIREMENTS_INSTALLER = REPO_ROOT / "install_requirements.bat"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DemoReleaseTests(unittest.TestCase):
    def test_model_matches_validated_release(self):
        self.assertTrue(MODEL.is_file())
        self.assertEqual(sha256(MODEL), MODEL_SHA256)

    def test_partial_dataset_manifest_is_complete(self):
        manifest = json.loads((DATASET_ROOT / "demo_subset_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["seed"], 2026)
        self.assertEqual(manifest["per_class"], 100)
        self.assertEqual(len(manifest["files"]), 300)
        self.assertEqual(Counter(item["class"] for item in manifest["files"]), Counter(apple=100, banana=100, orange=100))

        paths = []
        for item in manifest["files"]:
            path = (DATASET_ROOT / item["path"]).resolve()
            self.assertTrue(path.is_relative_to(DATASET_ROOT.resolve()))
            self.assertTrue(path.is_file(), item["path"])
            self.assertEqual(sha256(path), item["sha256"])
            paths.append(path)
        self.assertEqual(len(paths), len(set(paths)))

    def test_boot_service_precedes_daemon_and_waits_for_stock_overlay(self):
        daemon_unit = (FPGA_ROOT / "firmware" / "linux" / "budgetcuda.service").read_text(encoding="utf-8")
        fpga_unit = (FPGA_ROOT / "firmware" / "linux" / "budgetcuda-fpga.service").read_text(encoding="utf-8")
        fpga_setup = (FPGA_ROOT / "firmware" / "linux" / "budgetcuda-fpga-setup.sh").read_text(encoding="utf-8")
        overlay_helper = (FPGA_ROOT / "firmware" / "linux" / "budgetcuda_overlay.py").read_text(encoding="utf-8")
        afi_helper = (FPGA_ROOT / "firmware" / "linux" / "pynq_afi_hp0.py").read_text(encoding="utf-8")
        build_script = (FPGA_ROOT / "firmware" / "linux" / "build_server.sh").read_text(encoding="utf-8")
        installer = (REPO_ROOT / "install_budgetcuda_pynq.sh").read_text(encoding="utf-8")

        self.assertIn("Requires=budgetcuda-fpga.service", daemon_unit)
        self.assertIn("After=network-online.target budgetcuda-fpga.service", daemon_unit)
        self.assertIn("After=local-fs.target network-online.target", fpga_unit)
        self.assertIn("Before=budgetcuda.service", fpga_unit)
        self.assertIn('dmesg | grep -q "writing base.bin"', fpga_setup)
        self.assertIn('XILINX_XRT=/usr "$pynq_python"', fpga_setup)
        self.assertIn('"$afi_helper" --width 64', fpga_setup)
        self.assertIn('clock_mhz=${BCUDA_FCLK_MHZ:-100}', fpga_setup)
        self.assertIn('Environment=BCUDA_FCLK_MHZ=100', fpga_unit)
        self.assertIn("AFI_HP0_BASE = 0xF8008000", afi_helper)
        self.assertIn("ctypes.c_uint32.from_buffer", afi_helper)
        self.assertIn("Overlay(str(bitfile), download=True)", overlay_helper)
        self.assertIn('"PSDDR" not in overlay.mem_dict', overlay_helper)
        self.assertIn("-lxrt_coreutil", build_script)
        self.assertIn("bcuda_pynq_z2.hwh", installer)
        self.assertIn("pynq_afi_hp0.py", installer)
        self.assertIn("systemctl unmask bootpy.service", installer)
        self.assertIn("systemctl enable bootpy.service", installer)

    def test_host_requirements_exclude_cuda_specific_torch_packages(self):
        requirements = REQUIREMENTS.read_text(encoding="utf-8").lower().splitlines()
        packages = {
            line.split("==", 1)[0].strip()
            for line in requirements
            if line.strip() and not line.lstrip().startswith("#")
        }
        self.assertIn("onnx", packages)
        self.assertIn("onnxruntime", packages)
        self.assertIn("numpy", packages)
        self.assertNotIn("torch", packages)
        self.assertNotIn("torchvision", packages)

        installer = REQUIREMENTS_INSTALLER.read_text(encoding="utf-8").lower()
        self.assertIn('-r "%requirements%"', installer)
        self.assertIn('"setuptools<82"', installer)
        self.assertNotIn("pip install torch", installer)
        self.assertNotIn("pip install torchvision", installer)


if __name__ == "__main__":
    unittest.main()
