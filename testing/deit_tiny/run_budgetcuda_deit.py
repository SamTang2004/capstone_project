"""Run complete six-class DeiT inference through a BudgetCUDA board server."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FPGA"))

from budgetcuda import DeiTManifestRunner, Device, DiagnosticsSettings


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--manifest", type=Path, default=here / "artifacts" / "budgetcuda_w4a8" / "manifest.json")
    parser.add_argument("--address", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--production-repeat", type=int, default=0,
        help="Also measure this many steady-state runs without diagnostics",
    )
    parser.add_argument(
        "--compact", action="store_true",
        help="Omit detailed per-module and GEMM records from stdout",
    )
    parser.add_argument("--clock-hz", type=int, default=100_000_000)
    parser.add_argument(
        "--heatvit", action=argparse.BooleanOptionalAction, default=None,
        help="Enable or disable optional manifest token selectors",
    )
    parser.add_argument(
        "--cycle-report", type=Path,
        help="Also save the standard exact device cycle report as JSON",
    )
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.production_repeat < 0:
        parser.error("--production-repeat cannot be negative")
    diagnostics = DiagnosticsSettings(clock_hz=args.clock_hz)
    with DeiTManifestRunner(Device(args.address, args.port, args.timeout), args.manifest) as runner:
        load_started = perf_counter()
        runner.load(heatvit=args.heatvit)
        load_seconds = perf_counter() - load_started
        # Performance diagnostics are mandatory for FPGA model runs.  This
        # keeps module-level cycle consumption visible even when the caller
        # does not request a raw report file.
        results = [runner.run_image(
            args.image, diagnostics=diagnostics, heatvit=args.heatvit,
        ) for _ in range(args.repeat)]
        resident_weight_bytes = runner.resident_weight_bytes
        resident_buffer_count = runner.resident_buffer_count
        resident_total_bytes = runner.resident_total_bytes
        resident_total_buffer_count = runner.resident_total_buffer_count
        resident_descriptor_bytes = runner.resident_descriptor_bytes
        cycle_reports = [runner.cycle_report(result.diagnostics) for result in results]
        cycle_report = cycle_reports[-1]
        production_results = [runner.run_image(
            args.image, diagnostics=False, heatvit=args.heatvit,
        ) for _ in range(args.production_repeat)]
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
        "cycle_totals": cycle_report["totals"],
        "repeat_cycle_totals": [report["totals"] for report in cycle_reports],
        "production_repeat_seconds": [
            item.elapsed_seconds for item in production_results
        ],
        "production_stage_seconds": [
            dict(item.stage_seconds) for item in production_results
        ],
    }
    if not args.compact:
        summary["module_cycle_breakdown"] = cycle_report["module_cycle_breakdown"]
        summary["stage_cycle_breakdown"] = cycle_report["stage_cycle_breakdown"]
        summary["gemm_detail"] = cycle_report.get("gemm_detail", {})
    if args.cycle_report is not None:
        args.cycle_report.parent.mkdir(parents=True, exist_ok=True)
        args.cycle_report.write_text(json.dumps(cycle_report, indent=2), encoding="utf-8")
        summary["cycle_report"] = str(args.cycle_report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
