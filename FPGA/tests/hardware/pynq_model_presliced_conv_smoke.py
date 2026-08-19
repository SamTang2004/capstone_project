"""Protocol/XRT smoke test for one host-presliced INT8 Conv2D layer."""

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
    parser.add_argument("--legacy-weights", action="store_true")
    args = parser.parse_args()

    package = build_tiny_conv_relu_package(
        (1, 1, 3, 3),
        [[[[1, 0], [0, 1]]]],
        bias_values=[-10],
        name="board_presliced_conv_smoke",
        preslice_weights=not args.legacy_weights,
    )
    package.layers = package.layers[:1]
    package.output_index = package.layers[0].output

    model = Device(args.address, args.port, timeout=args.timeout).load_model(package)
    try:
        sentinel = np.asarray([0x11111111, 0x22222222, 0x33333333, 0x44444444], dtype=np.int32).reshape(1, 1, 2, 2)
        model.device.write(model.output_tensor, sentinel)
        actual = model.run(np.arange(1, 10, dtype=np.int8).reshape(1, 1, 3, 3))
        expected = np.asarray([-4, -2, 2, 4], dtype=np.int32).reshape(1, 1, 2, 2)
        if not np.array_equal(actual, expected):
            raw = model.device.read(model.output_tensor).reshape(-1)
            raise RuntimeError(
                "presliced Conv mismatch: "
                f"got {actual.reshape(-1).tolist()}, expected {expected.reshape(-1).tolist()}, "
                f"sentinel={[hex(int(value)) for value in sentinel.reshape(-1)]}, "
                f"physical output={[hex(int(value) & 0xffffffff) for value in raw]}"
            )
        mode = "legacy" if args.legacy_weights else "presliced"
        print(f"model {mode} Conv smoke passed: output={actual.reshape(-1).tolist()}")
    finally:
        model.close()


if __name__ == "__main__":
    main()
