"""Board probe for systolic GEMM row ordering through the production protocol."""

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

    graph = GraphBuilder("gemm_row_order")
    input_idx = graph.add_buffer("input", (4, 1), DType.INT8, usage=BufferUsage.ACTIVATION)
    weight_idx = graph.add_buffer("weight", (1, 1), DType.INT8, usage=BufferUsage.WEIGHT, data=[1])
    output_idx = graph.add_buffer("output", (4, 1), DType.INT32, usage=BufferUsage.OUTPUT)
    graph.gemm(input_idx, weight_idx, output_idx, m=4, n=1, k=1)
    package = graph.package(input_idx, output_idx)

    model = Device(args.address, args.port, timeout=30).load_model(package)
    try:
        probes = [
            np.asarray([1, 2, 4, 5], dtype=np.int8),
            np.asarray([1, 0, 0, 0], dtype=np.int8),
            np.asarray([0, 1, 0, 0], dtype=np.int8),
            np.asarray([0, 0, 1, 0], dtype=np.int8),
            np.asarray([0, 0, 0, 1], dtype=np.int8),
        ]
        failures = []
        for probe in probes:
            actual = model.run(probe.reshape(4, 1)).reshape(-1)
            print(f"GEMM row probe input={probe.tolist()} output={actual.tolist()}")
            if not np.array_equal(actual, probe.astype(np.int32)):
                failures.append((probe.tolist(), actual.tolist()))
        if failures:
            raise RuntimeError(f"GEMM row-order mismatches: {failures}")
        print("GEMM row-order smoke passed")
    finally:
        model.close()


if __name__ == "__main__":
    main()
