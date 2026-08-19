"""Board probe for Conv2D spatial-row ordering through the PL datapath."""

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
        (1, 1, 4, 4),
        [[[[1]]]],
        bias_values=[0],
        kernel_shape=(1, 1),
        name="board_conv_spatial_order_smoke",
    )
    package.layers = package.layers[:1]
    package.output_index = package.layers[0].output

    expected = np.arange(1, 17, dtype=np.int32).reshape(1, 1, 4, 4)
    model = Device(args.address, args.port, timeout=args.timeout).load_model(package)
    try:
        actual = model.run(expected.astype(np.int8))
        print(f"expected={expected.reshape(-1).tolist()}")
        print(f"actual={actual.reshape(-1).tolist()}")
        if not np.array_equal(actual, expected):
            raise RuntimeError("Conv2D spatial-order probe mismatch")
        print("Conv2D spatial-order probe passed")
    finally:
        model.close()


if __name__ == "__main__":
    main()
