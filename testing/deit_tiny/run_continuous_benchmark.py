"""Compare isolated DeiT latency with back-to-back resident-model throughput.

The current board protocol serializes complete graph executions.  This test
therefore measures continuous submission with a resident model, not concurrent
execution of multiple images inside the FPGA.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FPGA"))

from budgetcuda import DeiTManifestRunner, Device, DiagnosticsSettings


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile without extra dependencies."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--manifest", type=Path,
        default=here / "artifacts" / "heatvit_w4a8_balanced" / "manifest.json",
    )
    parser.add_argument("--address", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--clock-hz", type=int, default=100_000_000)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--heatvit", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")

    diagnostics = DiagnosticsSettings(clock_hz=args.clock_hz)
    with DeiTManifestRunner(
        Device(args.address, args.port, args.timeout), args.manifest,
    ) as runner:
        load_started = perf_counter()
        runner.load(heatvit=args.heatvit)
        load_seconds = perf_counter() - load_started

        isolated = runner.run_image(
            args.image, diagnostics=diagnostics, heatvit=args.heatvit,
        )
        isolated_report = runner.cycle_report(isolated.diagnostics)
        isolated_totals = isolated_report["totals"]

        warmup_seconds = []
        for _ in range(args.warmup):
            result = runner.run_image(
                args.image, diagnostics=False, heatvit=args.heatvit,
            )
            warmup_seconds.append(result.elapsed_seconds)

        stream_results = []
        stream_started = perf_counter()
        for _ in range(args.iterations):
            stream_results.append(runner.run_image(
                args.image, diagnostics=False, heatvit=args.heatvit,
            ))
        stream_wall_seconds = perf_counter() - stream_started

    host_latencies = [item.elapsed_seconds for item in stream_results]
    device_latencies = [dict(item.stage_seconds)["full_device_graph"]
                        for item in stream_results]
    representative_busy_seconds = isolated_totals["busy_cycles"] / args.clock_hz
    estimated_busy_seconds = representative_busy_seconds * args.iterations
    operations_per_image = isolated_totals["operations"]
    peak_gops = isolated_report["peak_operations_per_cycle"] * args.clock_hz / 1e9
    report = {
        "mode": "serialized_back_to_back_resident_graphs",
        "cross_image_hardware_overlap": False,
        "clock_hz": args.clock_hz,
        "iterations": args.iterations,
        "warmup_iterations": args.warmup,
        "model_load_seconds": load_seconds,
        "isolated_diagnostic": {
            "class_index": isolated.class_index,
            "class_name": isolated.class_name,
            "logits": list(isolated.logits),
            "host_seconds": isolated.elapsed_seconds,
            "device_graph_seconds": dict(isolated.stage_seconds)["full_device_graph"],
            "window_cycles": isolated_totals["window_cycles"],
            "busy_cycles": isolated_totals["busy_cycles"],
            "operations": operations_per_image,
            "busy_percent_within_graph": isolated_totals["busy_percent"],
            "busy_percent_of_host_wall": (
                100.0 * representative_busy_seconds / isolated.elapsed_seconds),
            "effective_gops": isolated_totals["effective_gops"],
            "arithmetic_peak_utilization_percent": (
                100.0 * isolated_totals["effective_gops"] / peak_gops),
        },
        "warmup_host_seconds": warmup_seconds,
        "continuous": {
            "first_class_index": stream_results[0].class_index,
            "first_class_name": stream_results[0].class_name,
            "first_logits": list(stream_results[0].logits),
            "last_logits": list(stream_results[-1].logits),
            "wall_seconds": stream_wall_seconds,
            "images_per_second": args.iterations / stream_wall_seconds,
            "device_graph_limited_images_per_second": (
                1.0 / statistics.fmean(device_latencies)),
            "mean_host_seconds": statistics.fmean(host_latencies),
            "p50_host_seconds": percentile(host_latencies, 0.50),
            "p95_host_seconds": percentile(host_latencies, 0.95),
            "mean_device_graph_seconds": statistics.fmean(device_latencies),
            "p50_device_graph_seconds": percentile(device_latencies, 0.50),
            "p95_device_graph_seconds": percentile(device_latencies, 0.95),
            "estimated_accelerator_busy_percent_of_wall": min(
                100.0, 100.0 * estimated_busy_seconds / stream_wall_seconds),
            "effective_gops_over_wall": (
                operations_per_image * args.iterations / stream_wall_seconds / 1e9),
            "arithmetic_peak_utilization_percent_over_wall": (
                100.0 * operations_per_image * args.iterations
                / stream_wall_seconds / 1e9 / peak_gops),
            "estimated_busy_cycles": isolated_totals["busy_cycles"] * args.iterations,
            "estimation_note": (
                "Uses the isolated run's deterministic busy cycles per graph "
                "over the production stream wall time; production runs disable "
                "diagnostic readback."
            ),
        },
    }
    encoded = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
