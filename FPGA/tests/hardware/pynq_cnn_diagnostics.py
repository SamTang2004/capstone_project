"""Run one BudgetCUDA CNN inference with firmware layer diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from budgetcuda import Device, DiagnosticsSettings, onnx_qdq_params
from budgetcuda.onnx_import import import_onnx_model
from pynq_full_cnn_inference import DEFAULT_MODEL


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--address", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--export", type=Path)
    parser.add_argument(
        "--no-reference", action="store_true",
        help="Skip the deterministic mock-backend layer comparison.",
    )
    args = parser.parse_args()

    package = import_onnx_model(args.model)
    input_buffer = package.buffers[package.input_index]
    values = np.random.default_rng(args.seed).integers(
        -8, 8, size=input_buffer.shape, dtype=np.int8
    )
    model = Device(args.address, args.port, timeout=args.timeout).load_model(package)
    try:
        result = model.run(values, diagnostics=DiagnosticsSettings())
    finally:
        model.close()

    report = result.diagnostics
    reference = None
    onnx_reference = None
    if not args.no_reference:
        reference_package = import_onnx_model(args.model)
        reference_model = Device.mock().load_model(reference_package)
        try:
            reference = reference_model.run(
                values, diagnostics=DiagnosticsSettings()
            ).diagnostics
        finally:
            reference_model.close()
        session = ort.InferenceSession(
            str(args.model), providers=["CPUExecutionProvider"]
        )
        onnx_reference = session.run(
            None, {session.get_inputs()[0].name: values}
        )[0]
    print(
        f"host_total_ms={report.host_total_ns / 1e6:.3f} "
        f"firmware_total_ms={report.firmware_total_ns / 1e6:.3f}"
    )
    for layer in report.layers:
        print(
            f"layer={layer.layer_index:02d} op={layer.op_name} "
            f"elapsed_ms={layer.elapsed_ns / 1e6:.3f} "
            f"roofline_util={layer.utilization_percent:.3f}% "
            f"compute_duty={layer.compute_duty_percent:.3f}% "
            f"loader_starve={layer.loader_starvation_percent:.3f}% "
            f"ddr_stall={layer.ddr_stall_percent:.3f}% "
            f"bottleneck={layer.likely_bottleneck!r} "
            f"hash={layer.output_hash:016x} "
            f"range=[{layer.output_min},{layer.output_max}] "
            f"saturation={layer.saturation_percent:.3f}%"
        )
    if reference is not None:
        divergence = report.first_divergence(reference)
        if divergence is None:
            print("accuracy_trace=all_recorded_layers_match")
        else:
            actual, expected = divergence
            print(
                f"accuracy_first_divergence=layer_{actual.layer_index} "
                f"op={actual.op_name} actual={actual.output_hash:016x} "
                f"reference={expected.output_hash:016x}"
            )
        output_qparams = onnx_qdq_params(args.model).output
        output_values = np.asarray(result.output, dtype=np.float32)
        dequantized = (
            output_values - output_qparams.zero_point
        ) * output_qparams.scale
        print(
            "onnx_output_comparison="
            f"max_abs_error={float(np.max(np.abs(dequantized - onnx_reference))):.8f} "
            f"board_class={int(np.argmax(output_values.reshape(-1)))} "
            f"onnx_class={int(np.argmax(onnx_reference.reshape(-1)))}"
        )
    if args.export is not None:
        report.export(args.export)
        print(f"exported={args.export}")


if __name__ == "__main__":
    main()
