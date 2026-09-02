"""Sweep TinyStories sparse scheduler granularity and control overhead."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "FPGA"))

from budgetcuda.stories import Llama2CCheckpoint  # noqa: E402
from budgetcuda.work_stealing_model import (  # noqa: E402
    evaluate_scheduler_overhead,
    write_scheduler_report,
)


def _integers(value: str) -> list[int]:
    try:
        result = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result or any(item < 0 for item in result):
        raise argparse.ArgumentTypeError("values must be non-negative integers")
    return result


def _chunks(value: str) -> list[int | None]:
    result = []
    for item in value.split(","):
        item = item.strip().lower()
        if item in ("whole", "none"):
            result.append(None)
        else:
            try:
                chunk = int(item)
            except ValueError as exc:
                raise argparse.ArgumentTypeError("chunks must be whole or positive integers") from exc
            if chunk <= 0:
                raise argparse.ArgumentTypeError("chunks must be whole or positive integers")
            result.append(chunk)
    if not result:
        raise argparse.ArgumentTypeError("at least one chunk size is required")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--significant-code", type=int, default=4)
    parser.add_argument("--chunks", type=_chunks, default=_chunks("whole,64,32,16"))
    parser.add_argument(
        "--overheads", type=_integers,
        default=_integers("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,64"),
    )
    parser.add_argument("--descriptor-bytes", type=int, default=16)
    parser.add_argument("--partial-commit-cycles", type=int, default=1)
    parser.add_argument("--barrier-cycles", type=int, default=2)
    parser.add_argument("--steal-batch", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = Llama2CCheckpoint(args.checkpoint, require_stories15m=True)
    report = evaluate_scheduler_overhead(
        checkpoint,
        workers=args.workers,
        significant_code=args.significant_code,
        chunk_sizes=args.chunks,
        overhead_values=args.overheads,
        descriptor_bytes=args.descriptor_bytes,
        partial_commit_cycles=args.partial_commit_cycles,
        operation_barrier_cycles=args.barrier_cycles,
        steal_batch=args.steal_batch,
    )
    json_path, markdown_path = write_scheduler_report(report, args.output)
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(markdown_path),
        "checkpoint_sha256": checkpoint.sha256,
    }, indent=2))


if __name__ == "__main__":
    main()
