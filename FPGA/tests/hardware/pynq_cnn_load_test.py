"""Run repeated CNN inference through one persistent BudgetCUDA model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from budgetcuda import Device, onnx_qdq_params
from budgetcuda.onnx_import import import_onnx_model
from pynq_full_cnn_inference import DEFAULT_MODEL


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--address", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    # Five truncating fixed-point stages are present in this deployed model.
    parser.add_argument("--ort-tolerance-lsb", type=float, default=5.0)
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")

    package = import_onnx_model(args.model)
    input_buffer = package.buffers[package.input_index]
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_params = onnx_qdq_params(args.model).output
    output_scale, output_zero_point = output_params.scale, output_params.zero_point
    tolerance = output_scale * args.ort_tolerance_lsb + 1e-6
    rng = np.random.default_rng(args.seed)

    device = Device(args.address, args.port, timeout=args.timeout)
    upload_start = time.perf_counter()
    board_model = device.load_model(package)
    upload_seconds = time.perf_counter() - upload_start
    mock_model = Device.mock().load_model(package)
    latencies: list[float] = []
    class_counts = np.zeros(package.buffers[package.output_index].shape[-1], dtype=np.int64)
    worst_onnx_error = 0.0
    try:
        for iteration in range(args.iterations):
            values = rng.integers(-32, 33, size=input_buffer.shape, dtype=np.int8)
            reference = session.run(None, {input_name: values})[0]
            expected = mock_model.run(values)
            inference_start = time.perf_counter()
            actual = board_model.run(values)
            latencies.append(time.perf_counter() - inference_start)

            if not np.array_equal(actual, expected):
                raise RuntimeError(
                    f"iteration {iteration} FPGA/software-device mismatch: "
                    f"FPGA={actual.reshape(-1).tolist()} mock={expected.reshape(-1).tolist()}"
                )
            actual_float = (actual.astype(np.float32) - output_zero_point) * output_scale
            error = float(np.max(np.abs(actual_float - reference)))
            worst_onnx_error = max(worst_onnx_error, error)
            if error > tolerance:
                raise RuntimeError(
                    f"iteration {iteration} exceeds ONNX tolerance: "
                    f"error={error:.8f} tolerance={tolerance:.8f}"
                )
            actual_class = int(np.argmax(actual.reshape(-1)))
            reference_class = int(np.argmax(reference.reshape(-1)))
            if actual_class != reference_class:
                raise RuntimeError(
                    f"iteration {iteration} classification mismatch: "
                    f"FPGA={actual_class} ONNX Runtime={reference_class}"
                )
            class_counts[actual_class] += 1
    finally:
        board_model.close()
        mock_model.close()

    total = sum(latencies)
    print(
        "CNN load test passed: "
        f"iterations={args.iterations} upload_s={upload_seconds:.6f} "
        f"total_inference_s={total:.6f} mean_ms={1000.0 * total / args.iterations:.3f} "
        f"p50_ms={1000.0 * percentile(latencies, 50):.3f} "
        f"p95_ms={1000.0 * percentile(latencies, 95):.3f} "
        f"max_ms={1000.0 * max(latencies):.3f} "
        f"worst_onnx_error={worst_onnx_error:.8f} "
        f"class_counts={class_counts.tolist()}"
    )


if __name__ == "__main__":
    main()
