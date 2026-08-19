"""Capture one per-layer diagnostics report from a deployed BudgetCUDA CNN."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from budgetcuda import Device, DiagnosticsSettings, import_onnx_model
from pynq_full_cnn_inference import DEFAULT_MODEL


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "diagnostics"
DEFAULT_CLOCK_HZ = 100_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--address", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--clock-hz", type=float, default=DEFAULT_CLOCK_HZ)
    args = parser.parse_args()
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")

    package = import_onnx_model(args.model)
    input_spec = package.buffers[package.input_index]
    rng = np.random.default_rng(args.seed)
    input_values = rng.integers(-32, 33, size=input_spec.shape, dtype=np.int8)
    settings = DiagnosticsSettings(
        timing=True,
        memory=True,
        operations=True,
        utilization=True,
        clock_hz=int(round(args.clock_hz)),
        peak_operations_per_cycle=512,
    )

    device = Device(args.address, args.port, timeout=args.timeout)
    upload_started = time.perf_counter_ns()
    model = device.load_model(package)
    upload_ns = time.perf_counter_ns() - upload_started
    try:
        for _ in range(args.warmup):
            model.run(input_values)
        result = model.run(input_values, diagnostics=settings)
    finally:
        model.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = result.diagnostics
    stem = args.output_dir / "fruit_cnn_diagnostic"
    report.export_csv(stem.with_suffix(".csv"))
    report.export_markdown(stem.with_suffix(".md"))
    report.export_text(stem.with_suffix(".txt"))

    print(
        "CNN diagnostics completed: "
        f"layers={len(report.layers)} upload_ms={upload_ns / 1e6:.3f} "
        f"firmware_ms={report.firmware_total_ns / 1e6:.3f} "
        f"host_execute_ms={report.host_execute_ns / 1e6:.3f} "
        f"host_total_ms={report.host_total_ns / 1e6:.3f} "
        f"output={np.asarray(result.output).reshape(-1).tolist()}"
    )
    print("Slowest layers:")
    for layer in sorted(report.layers, key=lambda item: item.elapsed_ns, reverse=True):
        share = 100.0 * layer.elapsed_ns / report.firmware_total_ns if report.firmware_total_ns else 0.0
        print(
            f"  {layer.layer_index:02d} {layer.op_name:<16} "
            f"time_ms={layer.elapsed_ns / 1e6:9.3f} share={share:6.2f}% "
            f"ops={layer.operations:12d} util={layer.utilization_percent:7.3f}% "
            f"working_set={layer.working_set_bytes}"
        )
    print(f"Reports: {stem.with_suffix('.csv')}, {stem.with_suffix('.md')}, {stem.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
