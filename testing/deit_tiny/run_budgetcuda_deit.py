"""Run complete six-class DeiT inference through a BudgetCUDA board server."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

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
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    with DeiTManifestRunner(Device(args.address, args.port, args.timeout), args.manifest) as runner:
        load_started = perf_counter()
        runner.load()
        load_seconds = perf_counter() - load_started
        results = [runner.run_image(args.image) for _ in range(args.repeat)]
        resident_weight_bytes = runner.resident_weight_bytes
        resident_buffer_count = runner.resident_buffer_count
        resident_total_bytes = runner.resident_total_bytes
        resident_total_buffer_count = runner.resident_total_buffer_count
        resident_descriptor_bytes = runner.resident_descriptor_bytes
    result = results[-1]
    print(json.dumps({
        "model_load_seconds": load_seconds,
        "resident_weight_bytes": resident_weight_bytes,
        "resident_buffer_count": resident_buffer_count,
        "resident_total_bytes": resident_total_bytes,
        "resident_total_buffer_count": resident_total_buffer_count,
        "resident_descriptor_bytes": resident_descriptor_bytes,
        "class_index": result.class_index,
        "class_name": result.class_name,
        "logits": result.logits,
        "elapsed_seconds": result.elapsed_seconds,
        "stages": dict(result.stage_seconds),
        "repeat_seconds": [item.elapsed_seconds for item in results],
    }, indent=2))


if __name__ == "__main__":
    main()
