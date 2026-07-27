from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda import Device, import_onnx_model, onnx_qdq_params, validate_model_package


ROOT = Path(__file__).resolve().parents[2]
MODEL = (
    ROOT
    / "train_model"
    / "models"
    / "fruit3_budgetcuda_int4"
    / "fruit3_budgetcuda_int4.onnx"
)


def test_trained_int4_fruit_model_matches_onnx_runtime():
    assert MODEL.is_file(), f"missing production test model: {MODEL}"
    package = import_onnx_model(MODEL)
    report = validate_model_package(package)
    assert report.arena_bytes_required < 64 * 1024 * 1024

    input_buffer = package.buffers[package.input_index]
    values = np.random.default_rng(2026).integers(
        -8,
        8,
        size=input_buffer.shape,
        dtype=np.int8,
    )
    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    reference = session.run(None, {session.get_inputs()[0].name: values})[0]

    model = Device.mock().load_model(package)
    try:
        actual = model.run(values)
    finally:
        model.close()

    qparams = onnx_qdq_params(MODEL).output
    actual_float = (actual.astype(np.float32) - qparams.zero_point) * qparams.scale
    assert actual.dtype == np.int8
    assert actual.shape == reference.shape
    assert int(np.argmax(actual)) == int(np.argmax(reference))
    assert float(np.max(np.abs(actual_float - reference))) <= qparams.scale * 4.0 + 1e-6
