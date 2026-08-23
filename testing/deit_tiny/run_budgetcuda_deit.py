"""Run complete six-class DeiT inference through a BudgetCUDA board server."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FPGA"))

from budgetcuda import DeiTManifestRunner, Device


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--manifest", type=Path, default=here / "artifacts" / "budgetcuda_w8a8" / "manifest.json")
    parser.add_argument("--address", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    result = DeiTManifestRunner(Device(args.address, args.port, args.timeout), args.manifest).run_image(args.image)
    print(json.dumps({
        "class_index": result.class_index,
        "class_name": result.class_name,
        "logits": result.logits,
        "elapsed_seconds": result.elapsed_seconds,
        "stages": dict(result.stage_seconds),
    }, indent=2))


if __name__ == "__main__":
    main()
