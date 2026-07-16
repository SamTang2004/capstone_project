"""End-to-end host -> PS daemon model upload and tiny Conv2D execution smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from budgetcuda import Device
from budgetcuda.compiler import build_tiny_conv_relu_package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    package = build_tiny_conv_relu_package(
        (1, 1, 3, 3),
        [[[[1, 0], [0, 1]]]],
        bias_values=[-10],
        name="board_upload_smoke",
    )
    device = Device(args.address, args.port, timeout=args.timeout)
    model = device.load_model(package)
    try:
        for host_buffer, device_tensor in zip(package.buffers, model.tensors):
            if host_buffer.data is None:
                continue
            actual = device.read(device_tensor).tobytes()
            if actual != host_buffer.data:
                raise RuntimeError(f"DRAM payload mismatch for {host_buffer.name}")

        result = model.run(np.arange(1, 10, dtype=np.int8).reshape(1, 1, 3, 3))
        expected = np.array([0, 0, 2, 4], dtype=np.int32).reshape(1, 1, 2, 2)
        if not np.array_equal(result, expected):
            raise RuntimeError(f"graph result mismatch: got {result.tolist()}, expected {expected.tolist()}")
        print(f"model load smoke passed: {len(model.tensors)} tensors and descriptor uploaded; output={result.reshape(-1).tolist()}")
    finally:
        model.close()


if __name__ == "__main__":
    main()
