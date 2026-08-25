"""Physical-board smoke test for the toggleable HeatViT token selector."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda import Device, build_heatvit_selector_package


def values(result) -> list[int]:
    return result.reshape(-1).tolist() if hasattr(result, "reshape") else list(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--mode", choices=("w8", "w4", "both"), default="both")
    args = parser.parse_args()

    sample = [
        9, 9,    # CLS: always retained
        1, 10,
        7, 20,   # highest score: retained
        3, 30,
        -2, 40,
    ]
    modes = (False, True) if args.mode == "both" else (args.mode == "w4",)
    board = Device(args.address, args.port, args.timeout)

    for int4 in modes:
        label = "W4A8" if int4 else "W8A8"
        package = build_heatvit_selector_package(
            5, 3, 2, [1, 0], weight_int4=int4,
            name=f"heatvit_selector_{label.lower()}",
        )
        reference = Device.mock().load_model(package)
        hardware = board.load_model(package)
        try:
            expected = values(reference.run(sample))
            actual = values(hardware.run(sample))
            if actual != expected:
                raise RuntimeError(
                    f"{label} selector mismatch: expected={expected}, actual={actual}")
            print(f"PASS: {label} HeatViT selector output={actual}")
        finally:
            reference.close()
            hardware.close()


if __name__ == "__main__":
    main()
