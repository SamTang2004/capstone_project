"""Run a complete quantized ONNX CNN through the BudgetCUDA board service."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from budgetcuda import Device, onnx_qdq_params, validate_model_package
from budgetcuda.compiler import ModelPackage
from budgetcuda.onnx_import import import_onnx_model


DEFAULT_MODEL = (
    ROOT
    / "train_model"
    / "models"
    / "fruit3_budgetcuda_int4"
    / "fruit3_budgetcuda_int4.onnx"
)


def run_package(device: Device, package: ModelPackage, input_values: np.ndarray) -> np.ndarray:
    model = device.load_model(package)
    try:
        return model.run(input_values)
    finally:
        model.close()


def diagnose_layers(device: Device, package: ModelPackage, input_values: np.ndarray) -> None:
    for index, layer in enumerate(package.layers):
        prefix = ModelPackage(
            buffers=package.buffers,
            layers=package.layers[: index + 1],
            input_index=package.input_index,
            output_index=layer.output,
            activation_arena_a=package.activation_arena_a,
            activation_arena_b=package.activation_arena_b,
            workspace_buffer=package.workspace_buffer,
            flags=package.flags,
        )
        expected = run_package(Device.mock(), prefix, input_values)
        actual = run_package(device, prefix, input_values)
        if not np.array_equal(actual, expected):
            difference = actual.astype(np.int64) - expected.astype(np.int64)
            mismatch_count = int(np.count_nonzero(difference))
            raise RuntimeError(
                f"first board divergence at layer {index} {layer.op_type.name}: "
                f"mismatches={mismatch_count}/{difference.size} "
                f"max_abs_delta={int(np.max(np.abs(difference)))} "
                f"FPGA={actual.reshape(-1)[:16].tolist()} "
                f"mock={expected.reshape(-1)[:16].tolist()}"
            )
        print(f"Layer {index:02d} {layer.op_type.name} passed ({actual.size} values)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--address", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--diagnose-layers", action="store_true")
    parser.add_argument("--ort-tolerance-lsb", type=float, default=4.0)
    args = parser.parse_args()

    package = import_onnx_model(args.model)
    preflight = validate_model_package(package)
    input_buffer = package.buffers[package.input_index]
    if input_buffer.dtype.name != "INT4":
        raise RuntimeError(f"expected an INT4 model input, got {input_buffer.dtype.name}")

    rng = np.random.default_rng(args.seed)
    input_values = rng.integers(-8, 8, size=input_buffer.shape, dtype=np.int8)

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    reference_start = time.perf_counter()
    reference = session.run(None, {session.get_inputs()[0].name: input_values})[0]
    reference_seconds = time.perf_counter() - reference_start

    device = Device(args.address, args.port, timeout=args.timeout)
    if args.diagnose_layers:
        diagnose_layers(device, package, input_values)

    mock_result = run_package(Device.mock(), package, input_values)
    upload_start = time.perf_counter()
    model = device.load_model(package)
    upload_seconds = time.perf_counter() - upload_start
    try:
        inference_start = time.perf_counter()
        actual = model.run(input_values)
        inference_seconds = time.perf_counter() - inference_start
    finally:
        model.close()

    if actual.dtype != np.int8 or actual.shape != reference.shape:
        raise RuntimeError(
            f"result metadata mismatch: expected FPGA INT4 container {reference.shape}, "
            f"got {actual.dtype} {actual.shape}"
        )
    if not np.array_equal(actual, mock_result):
        raise RuntimeError(
            f"FPGA/software-device mismatch: FPGA={actual.reshape(-1).tolist()} "
            f"mock={mock_result.reshape(-1).tolist()}"
        )
    output_params = onnx_qdq_params(args.model).output
    output_scale, output_zero_point = output_params.scale, output_params.zero_point
    actual_float = (actual.astype(np.float32) - output_zero_point) * output_scale
    max_error = float(np.max(np.abs(actual_float - reference)))
    # The deployed datapath truncates in each fixed-point requantizer and in
    # AveragePool, while ONNX QuantizeLinear rounds.  Exact FPGA-to-mock
    # equality above validates hardware; this bound covers accumulated model
    # quantization error in the externally dequantized ONNX output.
    if max_error > output_scale * args.ort_tolerance_lsb + 1e-6:
        raise RuntimeError(
            f"full CNN mismatch: FPGA INT4={actual.reshape(-1).tolist()} "
            f"FPGA float={actual_float.reshape(-1).tolist()} "
            f"reference={reference.reshape(-1).tolist()} max_error={max_error}"
        )

    predicted_class = int(np.argmax(actual.reshape(-1)))
    reference_class = int(np.argmax(reference.reshape(-1)))
    if predicted_class != reference_class:
        raise RuntimeError(
            f"classification mismatch: FPGA class={predicted_class}, "
            f"ONNX Runtime class={reference_class}"
        )
    print(
        "Full CNN inference passed: "
        f"layers={len(package.layers)} buffers={len(package.buffers)} "
        f"arena_bytes={preflight.arena_bytes_required} "
        f"int4_logits={actual.reshape(-1).tolist()} "
        f"float_logits={actual_float.reshape(-1).tolist()} "
        f"class={predicted_class} max_error={max_error:.8f} "
        f"ort_s={reference_seconds:.6f} upload_s={upload_seconds:.6f} "
        f"fpga_s={inference_seconds:.6f}"
    )


if __name__ == "__main__":
    main()
