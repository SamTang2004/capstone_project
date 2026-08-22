"""Probe every byte lane of the production HP0 read/write path with ReLU."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from budgetcuda import Device
from budgetcuda.compiler import GraphBuilder
from budgetcuda.dtypes import BufferUsage, DType


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()

    graph = GraphBuilder("activation_lane_smoke")
    input_idx = graph.add_buffer(
        "input", (32,), DType.INT8, usage=BufferUsage.ACTIVATION
    )
    output_idx = graph.add_buffer(
        "output", (32,), DType.INT8, usage=BufferUsage.OUTPUT
    )
    graph.relu(input_idx, output_idx)
    package = graph.package(input_idx, output_idx)

    values = np.arange(1, 33, dtype=np.int8)
    model = Device(args.address, args.port, timeout=30).load_model(package)
    try:
        model.device.write(model.output_tensor, np.full(32, -91, dtype=np.int8))
        actual = model.run(values)
        if not np.array_equal(actual, values):
            raise RuntimeError(
                "activation lane mismatch: "
                f"got={actual.reshape(-1).tolist()} expected={values.tolist()}"
            )
        print(f"Activation lane smoke passed: output={actual.tolist()}")
    finally:
        model.close()


if __name__ == "__main__":
    main()
