"""Probe every 32-bit output lane of the production HP0 GEMM path."""

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

    graph = GraphBuilder("gemm_lane_smoke")
    input_idx = graph.add_buffer("input", (4, 4), DType.INT8, usage=BufferUsage.ACTIVATION)
    weight_idx = graph.add_buffer(
        "weight",
        (4, 4),
        DType.INT8,
        usage=BufferUsage.WEIGHT,
        data=np.eye(4, dtype=np.int8).reshape(-1).tolist(),
    )
    output_idx = graph.add_buffer("output", (4, 4), DType.INT32, usage=BufferUsage.OUTPUT)
    graph.gemm(input_idx, weight_idx, output_idx, m=4, n=4, k=4)
    package = graph.package(input_idx, output_idx)

    values = np.arange(1, 17, dtype=np.int8).reshape(4, 4)
    model = Device(args.address, args.port, timeout=30).load_model(package)
    try:
        sentinel = np.full((4, 4), 0x22222222, dtype=np.int32)
        model.device.write(model.output_tensor, sentinel)
        actual = model.run(values)
        if not np.array_equal(actual, values.astype(np.int32)):
            raise RuntimeError(
                "GEMM lane mismatch: "
                f"got={actual.reshape(-1).tolist()} "
                f"expected={values.reshape(-1).tolist()}"
            )
        print(f"GEMM lane smoke passed: output={actual.reshape(-1).tolist()}")
    finally:
        model.close()


if __name__ == "__main__":
    main()
