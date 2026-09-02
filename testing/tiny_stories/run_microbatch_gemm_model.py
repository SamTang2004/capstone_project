"""Sweep significance-screen overhead and tagged GEMV-to-GEMM packing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "FPGA"))

from budgetcuda.microbatch_gemm_model import (  # noqa: E402
    evaluate_microbatch_sweep,
    write_microbatch_report,
)
from budgetcuda.stories import Llama2CCheckpoint  # noqa: E402


def _integers(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return result


def _floats(value: str) -> list[float]:
    try:
        result = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("values must be non-negative")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "--batch-activation-factors", type=_floats,
        default=_floats("0.05,0.1,0.25,0.5,1,1,1,1"),
    )
    parser.add_argument("--activation-density", type=float, default=0.5)
    parser.add_argument("--significant-code", type=int, default=4)
    parser.add_argument("--chunks", type=_integers, default=_integers("1,4,8,16,32,64"))
    parser.add_argument("--predicate-lanes", type=_integers, default=_integers("1,2,4,8,16"))
    parser.add_argument("--predicate-cycles", type=int, default=1)
    parser.add_argument("--tag-enqueue-cycles", type=int, default=1)
    parser.add_argument("--enqueue-lanes", type=int, default=1)
    parser.add_argument("--matrix-multipliers", type=int, default=2)
    parser.add_argument("--gemm-rows", type=int, default=8)
    parser.add_argument("--gemm-launch-cycles", type=int, default=2)
    parser.add_argument("--tag-route-cycles", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = Llama2CCheckpoint(args.checkpoint, require_stories15m=True)
    report = evaluate_microbatch_sweep(
        checkpoint,
        batch_activation_factors=args.batch_activation_factors,
        activation_density=args.activation_density,
        significant_code=args.significant_code,
        chunk_sizes=args.chunks,
        predicate_lanes=args.predicate_lanes,
        predicate_cycles=args.predicate_cycles,
        tag_enqueue_cycles=args.tag_enqueue_cycles,
        enqueue_lanes=args.enqueue_lanes,
        matrix_multipliers=args.matrix_multipliers,
        gemm_rows=args.gemm_rows,
        gemm_launch_cycles=args.gemm_launch_cycles,
        tag_route_cycles=args.tag_route_cycles,
        seed=args.seed,
    )
    json_path, markdown_path = write_microbatch_report(report, args.output)
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(markdown_path),
        "samples": len(report["results"]),
    }, indent=2))


if __name__ == "__main__":
    main()
