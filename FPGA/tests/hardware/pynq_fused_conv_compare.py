"""Compare the fused first CNN block with its standalone PL execution."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from budgetcuda import Device
from budgetcuda.compiler import ModelPackage
from budgetcuda.onnx_import import import_onnx_model


DEFAULT_MODEL = (
    ROOT
    / "train_model"
    / "models"
    / "fruit3_budgetcuda_int4"
    / "fruit3_budgetcuda_int4.onnx"
)


def prefix(package: ModelPackage, layer_count: int) -> ModelPackage:
    layer = package.layers[layer_count - 1]
    return ModelPackage(
        buffers=package.buffers,
        layers=package.layers[:layer_count],
        input_index=package.input_index,
        output_index=layer.output,
        activation_arena_a=package.activation_arena_a,
        activation_arena_b=package.activation_arena_b,
        workspace_buffer=package.workspace_buffer,
        flags=package.flags,
    )


def run(device: Device, package: ModelPackage, values: np.ndarray) -> np.ndarray:
    model = device.load_model(package)
    try:
        return model.run(values)
    finally:
        model.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--address", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    package = import_onnx_model(args.model)
    input_buffer = package.buffers[package.input_index]
    values = np.random.default_rng(args.seed).integers(
        -8, 8, size=input_buffer.shape, dtype=np.int8
    )
    board = Device(args.address, args.port, timeout=args.timeout)
    standalone = run(board, prefix(package, 2), values)
    fused = run(board, prefix(package, 3), values)
    expected = np.maximum(standalone, 0).astype(np.int8)
    difference = fused.astype(np.int16) - expected.astype(np.int16)
    mismatch = np.flatnonzero(difference)

    print(
        f"standalone_range={int(standalone.min())}..{int(standalone.max())} "
        f"fused_range={int(fused.min())}..{int(fused.max())} "
        f"mismatches={mismatch.size}/{difference.size} "
        f"max_abs_delta={int(np.max(np.abs(difference)))}"
    )
    for channel in range(fused.shape[1]):
        channel_values = fused[0, channel].reshape(-1)
        matches = [
            int(np.count_nonzero(channel_values == expected[0, candidate].reshape(-1)))
            for candidate in range(expected.shape[1])
        ]
        best_channel = int(np.argmax(matches))
        own_matches = matches[channel]
        print(
            f"channel={channel} own_matches={own_matches}/{channel_values.size} "
            f"best_expected_channel={best_channel} "
            f"best_matches={matches[best_channel]}/{channel_values.size}"
        )
    for index in mismatch[:16]:
        coordinate = np.unravel_index(index, fused.shape)
        print(
            f"index={index} coordinate={coordinate} "
            f"standalone={int(standalone.flat[index])} "
            f"expected={int(expected.flat[index])} "
            f"fused={int(fused.flat[index])}"
        )
    if mismatch.size:
        raise SystemExit(1)
    print("Fused Conv2D board comparison passed")


if __name__ == "__main__":
    main()
