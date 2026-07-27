"""Evaluate the trained INT8 fruit CNN on Fruits-360 and PYNQ-Z2."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import random
import sys
import time

import numpy as np
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[2]
TRAIN_MODEL = ROOT / "train_model"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TRAIN_MODEL))

from budgetcuda import Device, load_quantized_image, onnx_qdq_params
from budgetcuda.onnx_import import import_onnx_model
from paths import DATASET_ROOT
from qat_model import CLASS_NAMES, IMAGE_SIZE, GroupedFruitDataset
from pynq_full_cnn_inference import DEFAULT_MODEL

DEMO_DATASET_ROOT = ROOT / "demo" / "dataset" / "fruits-360-100x100"
DEFAULT_DATASET_ROOT = DEMO_DATASET_ROOT if DEMO_DATASET_ROOT.exists() else DATASET_ROOT


def preprocess(path: Path, input_shape, input_params) -> np.ndarray:
    return load_quantized_image(path, input_shape, input_params, layout="nchw")


def format_matrix(matrix: np.ndarray) -> str:
    rows = ["true\\pred " + " ".join(f"{name:>7}" for name in CLASS_NAMES)]
    rows.extend(f"{CLASS_NAMES[index]:>9} " + " ".join(f"{int(value):7d}" for value in row) for index, row in enumerate(matrix))
    return "\n".join(rows)


def evaluate_onnx(session, input_name: str, dataset, input_shape, input_params, label: str):
    matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    predictions: dict[Path, int] = {}
    start = time.perf_counter()
    for index, (path, target, _) in enumerate(dataset.samples, 1):
        values = preprocess(path, input_shape, input_params)
        prediction = int(np.argmax(session.run(None, {input_name: values})[0].reshape(-1)))
        matrix[target, prediction] += 1
        predictions[path] = prediction
        if index % 1000 == 0:
            print(f"ONNX {label}: {index}/{len(dataset)}", flush=True)
    elapsed = time.perf_counter() - start
    accuracy = float(np.trace(matrix) / matrix.sum())
    print(f"ONNX {label} accuracy={accuracy:.6f} images={matrix.sum()} elapsed_s={elapsed:.3f}")
    print(format_matrix(matrix))
    return predictions


def balanced_samples(dataset, per_class: int, seed: int):
    grouped = defaultdict(list)
    for sample in dataset.samples:
        grouped[sample[1]].append(sample)
    rng = random.Random(seed)
    selected = []
    for target in range(len(CLASS_NAMES)):
        candidates = grouped[target]
        if len(candidates) < per_class:
            raise RuntimeError(f"class {CLASS_NAMES[target]} has only {len(candidates)} test images")
        selected.extend(rng.sample(candidates, per_class))
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--address", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--board-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    if args.board_per_class <= 0:
        raise ValueError("board-per-class must be positive")

    test_dataset = GroupedFruitDataset(args.dataset_root / "Test")
    input_params = onnx_qdq_params(args.model).input
    package = import_onnx_model(args.model)
    input_shape = package.buffers[package.input_index].shape
    if input_shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"fruit model input shape is {input_shape}, expected (1, 3, {IMAGE_SIZE}, {IMAGE_SIZE})")
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    if not args.skip_train:
        train_dataset = GroupedFruitDataset(args.dataset_root / "Training")
        evaluate_onnx(session, input_name, train_dataset, input_shape, input_params, "train")
    test_predictions = evaluate_onnx(session, input_name, test_dataset, input_shape, input_params, "test")

    selected = balanced_samples(test_dataset, args.board_per_class, args.seed)
    board_matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    agreement = 0
    latencies = []
    device = Device(args.address, args.port, timeout=args.timeout)
    upload_start = time.perf_counter()
    model = device.load_model(package)
    upload_seconds = time.perf_counter() - upload_start
    try:
        for index, (path, target, _) in enumerate(selected, 1):
            values = preprocess(path, input_shape, input_params)
            start = time.perf_counter()
            logits = model.run(values)
            latencies.append(time.perf_counter() - start)
            prediction = int(np.argmax(logits.reshape(-1)))
            board_matrix[target, prediction] += 1
            agreement += prediction == test_predictions[path]
            if index % 25 == 0:
                print(f"FPGA test: {index}/{len(selected)}", flush=True)
    finally:
        model.close()

    board_accuracy = float(np.trace(board_matrix) / board_matrix.sum())
    print(
        f"FPGA balanced-test accuracy={board_accuracy:.6f} images={board_matrix.sum()} "
        f"onnx_agreement={agreement / len(selected):.6f} upload_s={upload_seconds:.6f} "
        f"mean_ms={1000.0 * float(np.mean(latencies)):.3f} "
        f"p95_ms={1000.0 * float(np.percentile(latencies, 95)):.3f} "
        f"max_ms={1000.0 * max(latencies):.3f}"
    )
    print(format_matrix(board_matrix))


if __name__ == "__main__":
    main()
