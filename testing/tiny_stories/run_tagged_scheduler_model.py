"""Measure batch-tag overhead in the TinyStories sparse work-stealing model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "FPGA"))

from budgetcuda.stories import Llama2CCheckpoint  # noqa: E402
from budgetcuda.work_stealing_model import (  # noqa: E402
    evaluate_tagged_batch_overhead,
    write_tagged_batch_report,
)


def _integers(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("values must be non-negative integers")
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
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-active-factors", type=_floats, default=_floats("0.05,1.0"))
    parser.add_argument("--significant-code", type=int, default=4)
    parser.add_argument("--k-chunk", type=int)
    parser.add_argument(
        "--tag-lookup-cycles", type=_integers,
        default=_integers("0,1,2,4,8,12,16,24,32,33,34,35,36,37,38,39,40,48,64"),
    )
    parser.add_argument("--task-dispatch-cycles", type=int, default=1)
    parser.add_argument("--steal-probe-cycles", type=int, default=4)
    parser.add_argument("--steal-transfer-cycles", type=int, default=4)
    parser.add_argument("--partial-commit-cycles", type=int, default=1)
    parser.add_argument("--barrier-cycles", type=int, default=2)
    parser.add_argument("--steal-batch", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = Llama2CCheckpoint(args.checkpoint, require_stories15m=True)
    report = evaluate_tagged_batch_overhead(
        checkpoint,
        workers=args.workers,
        batch_active_factors=args.batch_active_factors,
        significant_code=args.significant_code,
        k_chunk_steps=args.k_chunk,
        tag_lookup_values=args.tag_lookup_cycles,
        task_dispatch_cycles=args.task_dispatch_cycles,
        steal_probe_cycles=args.steal_probe_cycles,
        steal_transfer_cycles=args.steal_transfer_cycles,
        partial_commit_cycles=args.partial_commit_cycles,
        operation_barrier_cycles=args.barrier_cycles,
        steal_batch=args.steal_batch,
    )
    json_path, markdown_path = write_tagged_batch_report(report, args.output)
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(markdown_path),
        "tag_bits": report["tag_bits"],
        "break_even_tag_lookup_cycles": report["max_sampled_break_even_tag_lookup_cycles"],
    }, indent=2))


if __name__ == "__main__":
    main()
