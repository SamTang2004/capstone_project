import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda import Device, DiagnosticResult, DiagnosticsSettings
from budgetcuda.compiler import build_tiny_conv_relu_package
from budgetcuda.diagnostics import parse_diagnostics


class DiagnosticsTest(unittest.TestCase):
    def test_settings_validate_dependencies_and_report_size(self):
        with self.assertRaisesRegex(ValueError, "require timing"):
            DiagnosticsSettings(timing=False, utilization=True)
        with self.assertRaisesRegex(ValueError, "At least one"):
            DiagnosticsSettings(timing=False, memory=False, operations=False, utilization=False)
        settings = DiagnosticsSettings()
        self.assertEqual(settings.buffer_bytes(2), 48 + 2 * 144)
        with self.assertRaisesRegex(ValueError, "daemon may not support"):
            parse_diagnostics(settings.initial_buffer(2), settings)

    def test_mock_model_run_returns_output_and_layer_curves(self):
        package = build_tiny_conv_relu_package(
            (1, 1, 3, 3),
            [[[[1, 0], [0, 1]]]],
            bias_values=[-10],
        )
        model = Device.mock().load_model(package)
        try:
            result = model.run(
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                diagnostics=DiagnosticsSettings(),
            )
        finally:
            model.close()

        self.assertIsInstance(result, DiagnosticResult)
        values = result.output.reshape(-1).tolist() if hasattr(result.output, "reshape") else result.output
        self.assertEqual(values, [0, 0, 2, 4])
        self.assertEqual(len(result.diagnostics.layers), 2)
        self.assertEqual([layer.op_name for layer in result.diagnostics.layers], ["CONV2D_GEMM", "ACTIVATION"])
        self.assertEqual(len(result.diagnostics.time_curve), 2)
        self.assertEqual(len(result.diagnostics.memory_curve), 2)
        self.assertEqual(len(result.diagnostics.usage_curve), 2)
        self.assertEqual(len(result.diagnostics.utilization_curve), 2)
        self.assertGreater(result.diagnostics.layers[0].operations, 0)
        self.assertGreater(result.diagnostics.layers[0].weight_bytes, 0)
        self.assertGreater(result.diagnostics.host_total_ns, 0)

    def test_selective_reports_and_exports(self):
        package = build_tiny_conv_relu_package((1, 1, 2, 2), [[[[1]]]], kernel_shape=(1, 1))
        model = Device.mock().load_model(package)
        settings = DiagnosticsSettings(timing=False, memory=True, operations=False, utilization=False)
        try:
            report = model.run([1, 2, 3, 4], diagnostics=settings).diagnostics
        finally:
            model.close()
        self.assertTrue(all(value == 0 for value in report.time_curve))
        self.assertTrue(all(value == 0 for value in report.operation_curve))
        self.assertTrue(any(value > 0 for value in report.memory_curve))
        self.assertIn("layer_index,op_type,op_name", report.to_csv())
        self.assertIn("# BudgetCUDA Run Diagnostics", report.to_markdown())
        self.assertIn("BudgetCUDA Run Diagnostics", report.to_text())
        self.assertEqual(report.to_txt(), report.to_text())
        with tempfile.TemporaryDirectory() as directory:
            path = report.export(Path(directory) / "run.md")
            self.assertTrue(path.read_text(encoding="utf-8").startswith("# BudgetCUDA"))
            csv_path = report.export_csv(Path(directory) / "run.csv")
            self.assertTrue(csv_path.read_text(encoding="utf-8").startswith("layer_index"))


if __name__ == "__main__":
    unittest.main()
