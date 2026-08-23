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
    parser.add_argument(
        "--cycle-report", type=Path,
        help="Write exact per-layer 64-bit device cycle counters as JSON",
    )
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    with DeiTManifestRunner(Device(args.address, args.port, args.timeout), args.manifest) as runner:
        load_started = perf_counter()
        runner.load()
        load_seconds = perf_counter() - load_started
        results = [
            runner.run_image(args.image, diagnostics=args.cycle_report is not None)
            for _ in range(args.repeat)
        ]
        resident_weight_bytes = runner.resident_weight_bytes
        resident_buffer_count = runner.resident_buffer_count
        resident_total_bytes = runner.resident_total_bytes
        resident_total_buffer_count = runner.resident_total_buffer_count
        resident_descriptor_bytes = runner.resident_descriptor_bytes
        cycle_report = (
            runner.cycle_report(results[-1].diagnostics)
            if results[-1].diagnostics is not None else None
        )
    result = results[-1]
    summary = {
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
    }
    if cycle_report is not None:
        args.cycle_report.parent.mkdir(parents=True, exist_ok=True)
        args.cycle_report.write_text(json.dumps(cycle_report, indent=2), encoding="utf-8")
        summary["cycle_report"] = str(args.cycle_report)
        summary["cycle_totals"] = cycle_report["totals"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
