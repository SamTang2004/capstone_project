"""Plot windowed memory-path utilization from saturation simulation traces."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "budgetcuda-matplotlib"))
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    "req_accept",
    "read_beat",
    "out_beat",
    "lookup_pending",
    "lookup_result_pending",
    "fill_active",
    "deliver_active",
    "read_pending",
    "prepared_valid",
)


def read_trace(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Trace contains no samples: {path}")
    columns = set(rows[0])
    result: dict[str, np.ndarray] = {}
    for name in ("cycle", *METRICS, "busy", "request_count", "out_bytes"):
        if name in columns:
            result[name] = np.asarray([int(row[name]) for row in rows], dtype=np.float64)
        else:
            result[name] = np.zeros(len(rows), dtype=np.float64)
    if "out_bytes" not in columns:
        result["out_bytes"] = result["out_beat"].copy()
    return result


def windows(trace: dict[str, np.ndarray], width: int) -> list[dict[str, float]]:
    count = len(trace["cycle"])
    samples: list[dict[str, float]] = []
    for start in range(0, count, width):
        stop = min(start + width, count)
        sample = {
            "start_cycle": int(trace["cycle"][start]),
            "end_cycle": int(trace["cycle"][stop - 1]),
        }
        for name in METRICS:
            sample[name] = float(np.mean(trace[name][start:stop]))
        sample["busy"] = float(np.mean(trace["busy"][start:stop]))
        sample["request_count"] = float(np.mean(trace["request_count"][start:stop]))
        samples.append(sample)
    return samples


def summarize(trace: dict[str, np.ndarray]) -> dict[str, float]:
    cycles = len(trace["cycle"])
    return {
        "cycles": cycles,
        "requests": int(np.sum(trace["req_accept"])),
        "read_payload_beats": int(np.sum(trace["read_beat"])),
        "output_bytes": int(np.sum(trace["out_bytes"])),
        "read_bus_utilization_percent": 100.0 * float(np.mean(trace["read_beat"])),
        "output_stream_utilization_percent": 100.0 * float(np.mean(trace["out_beat"])),
        "request_accept_utilization_percent": 100.0 * float(np.mean(trace["req_accept"])),
        "busy_percent": 100.0 * float(np.mean(trace["busy"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    parser.add_argument("--vector", type=Path)
    parser.add_argument("--wide", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=64)
    args = parser.parse_args()
    if args.window <= 0:
        raise ValueError("window must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = {
        "baseline": read_trace(args.baseline),
        "queued_lookup": read_trace(args.optimized),
    }
    if args.vector is not None:
        traces["vector16"] = read_trace(args.vector)
    if args.wide is not None:
        traces["wide32"] = read_trace(args.wide)
    windowed = {name: windows(trace, args.window) for name, trace in traces.items()}
    summary = {name: summarize(trace) for name, trace in traces.items()}
    baseline_cycles = summary["baseline"]["cycles"]
    optimized_cycles = summary["queued_lookup"]["cycles"]
    summary["cycle_reduction_percent"] = 100.0 * (baseline_cycles - optimized_cycles) / baseline_cycles
    if "vector16" in summary:
        vector_cycles = summary["vector16"]["cycles"]
        summary["vector_cycle_reduction_vs_scalar_percent"] = (
            100.0 * (optimized_cycles - vector_cycles) / optimized_cycles
        )
    if "wide32" in summary:
        wide_cycles = summary["wide32"]["cycles"]
        vector_reference = summary.get("vector16", summary["queued_lookup"])["cycles"]
        summary["wide_cycle_reduction_vs_vector_percent"] = (
            100.0 * (vector_reference - wide_cycles) / vector_reference
        )

    with (args.output_dir / "utilization_windows.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["configuration", "start_cycle", "end_cycle", *METRICS, "busy", "request_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for configuration, samples in windowed.items():
            for sample in samples:
                writer.writerow({"configuration": configuration, **sample})
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    colors = {
        "baseline": "#7a7f87",
        "queued_lookup": "#087e8b",
        "vector16": "#d95f02",
        "wide32": "#6a3d9a",
    }
    for configuration, samples in windowed.items():
        x = np.asarray([(sample["start_cycle"] + sample["end_cycle"]) / 2 for sample in samples])
        axes[0, 0].plot(x, 100 * np.asarray([sample["read_beat"] for sample in samples]),
                        label=configuration, color=colors[configuration], linewidth=1.8)
        axes[0, 1].plot(x, 100 * np.asarray([sample["out_beat"] for sample in samples]),
                        label=configuration, color=colors[configuration], linewidth=1.8)
        axes[1, 0].plot(x, 100 * np.asarray([sample["req_accept"] for sample in samples]),
                        label=configuration, color=colors[configuration], linewidth=1.8)
    optimized_samples = windowed["queued_lookup"]
    x = np.asarray([(sample["start_cycle"] + sample["end_cycle"]) / 2 for sample in optimized_samples])
    for metric, color in (
        ("fill_active", "#d95f02"),
        ("deliver_active", "#1b9e77"),
        ("lookup_pending", "#7570b3"),
        ("prepared_valid", "#e7298a"),
    ):
        axes[1, 1].plot(x, 100 * np.asarray([sample[metric] for sample in optimized_samples]),
                        label=metric, color=color, linewidth=1.5)

    axes[0, 0].set_title("DDR Read Payload Occupancy")
    axes[0, 1].set_title("Output Response Occupancy")
    axes[1, 0].set_title("Request Acceptance Occupancy")
    axes[1, 1].set_title("Queued Design Stage Occupancy")
    for axis in axes.flat:
        axis.set_xlabel("Cycle")
        axis.set_ylabel("Window utilization (%)")
        axis.set_ylim(0, 105)
        axis.grid(True, color="#d9dde2", linewidth=0.6)
        axis.legend(frameon=False)
    figure.suptitle(f"Memory IO Load-Balance Stress, {args.window}-cycle windows")
    figure.savefig(args.output_dir / "utilization_curves.png", dpi=180)
    figure.savefig(args.output_dir / "utilization_curves.svg")
    plt.close(figure)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_dir / 'utilization_curves.png'}")


if __name__ == "__main__":
    main()
