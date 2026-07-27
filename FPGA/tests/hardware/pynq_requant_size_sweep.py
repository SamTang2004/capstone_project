"""Find the first packed-INT4 DDR requantization size that fails on PYNQ-Z2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from budgetcuda import Device
from budgetcuda.compiler import HostBuffer, Layer, ModelPackage
from budgetcuda.dtypes import BufferUsage, DType, OpType


def build_package(element_count: int) -> ModelPackage:
    return ModelPackage(
        buffers=[
            HostBuffer("input", (element_count,), DType.INT32),
            HostBuffer.from_values(
                "params",
                (4,),
                DType.INT32,
                [3, 2, 0, 0],
                usage=BufferUsage.WEIGHT,
            ),
            HostBuffer(
                "output",
                (element_count,),
                DType.INT4,
                usage=BufferUsage.OUTPUT,
            ),
        ],
        layers=[
            Layer(
                OpType.REQUANTIZE,
                input0=0,
                output=2,
                scale=1,
            )
        ],
        input_index=0,
        output_index=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--sizes",
        default="1,3,8,16,64,256,1024,4096,16384,32768,65536",
    )
    args = parser.parse_args()

    sizes = [int(value) for value in args.sizes.split(",")]
    board = Device(args.address, args.port, timeout=args.timeout)
    rng = np.random.default_rng(2026)
    for element_count in sizes:
        package = build_package(element_count)
        values = rng.integers(
            -(1 << 20),
            1 << 20,
            size=(element_count,),
            dtype=np.int32,
        )
        expected_model = Device.mock().load_model(package)
        board_model = board.load_model(package)
        try:
            expected = expected_model.run(values)
            started = time.perf_counter()
            actual = board_model.run(values)
            elapsed = time.perf_counter() - started
        finally:
            expected_model.close()
            board_model.close()
        if not np.array_equal(actual, expected):
            difference = actual.astype(np.int16) - expected.astype(np.int16)
            raise RuntimeError(
                f"requantize mismatch at size={element_count}: "
                f"mismatches={np.count_nonzero(difference)} "
                f"max_abs_delta={np.max(np.abs(difference))}"
            )
        print(
            f"requantize size={element_count} passed "
            f"elapsed_ms={elapsed * 1000.0:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
