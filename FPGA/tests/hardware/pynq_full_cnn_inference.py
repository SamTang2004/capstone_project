"""Run a complete quantized ONNX CNN through the BudgetCUDA board service."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from budgetcuda import Device
from budgetcuda.compiler import ModelPackage
from budgetcuda.onnx_import import import_onnx_model


DEFAULT_MODEL = (
    ROOT
    / "train_model"
    / "models"
    / "fruit3_budgetcuda_qat_smoke"
    / "fruit3_budgetcuda_int8.onnx"
)


def output_quantization(model_path: Path) -> tuple[float, int]:
    model = onnx.load(str(model_path))
    producers = {output: node for node in model.graph.node for output in node.output}
    initializers = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    output_name = model.graph.output[0].name
    node = producers.get(output_name)
    if node is None or node.op_type != "DequantizeLinear" or len(node.input) < 3:
        raise RuntimeError("model output must be produced by DequantizeLinear")
    try:
        scale = float(initializers[node.input[1]].reshape(-1)[0])
        zero_point = int(initializers[node.input[2]].reshape(-1)[0])
    except (KeyError, IndexError) as exc:
        raise RuntimeError("output quantization parameters must be scalar initializers") from exc
    return scale, zero_point


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
    input_buffer = package.buffers[package.input_index]
    if input_buffer.dtype.name != "INT8":
        raise RuntimeError(f"expected an INT8 model input, got {input_buffer.dtype.name}")

    rng = np.random.default_rng(args.seed)
    input_values = rng.integers(-32, 33, size=input_buffer.shape, dtype=np.int8)

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
            f"result metadata mismatch: expected FPGA int8 {reference.shape}, "
            f"got {actual.dtype} {actual.shape}"
        )
    if not np.array_equal(actual, mock_result):
        raise RuntimeError(
            f"FPGA/software-device mismatch: FPGA={actual.reshape(-1).tolist()} "
            f"mock={mock_result.reshape(-1).tolist()}"
        )
    output_scale, output_zero_point = output_quantization(args.model)
    actual_float = (actual.astype(np.float32) - output_zero_point) * output_scale
    max_error = float(np.max(np.abs(actual_float - reference)))
    # The deployed datapath truncates in each fixed-point requantizer and in
    # AveragePool, while ONNX QuantizeLinear rounds.  Exact FPGA-to-mock
    # equality above validates hardware; this bound covers accumulated model
    # quantization error in the externally dequantized ONNX output.
    if max_error > output_scale * args.ort_tolerance_lsb + 1e-6:
        raise RuntimeError(
            f"full CNN mismatch: FPGA int8={actual.reshape(-1).tolist()} "
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
        f"bytes={sum(buffer.size_bytes for buffer in package.buffers)} "
        f"int8_logits={actual.reshape(-1).tolist()} "
        f"float_logits={actual_float.reshape(-1).tolist()} "
        f"class={predicted_class} max_error={max_error:.8f} "
        f"ort_s={reference_seconds:.6f} upload_s={upload_seconds:.6f} "
        f"fpga_s={inference_seconds:.6f}"
    )


if __name__ == "__main__":
    main()
