"""Measure TinyStories accuracy after tile-K significance screening."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "FPGA"))

from budgetcuda.stories import Llama2CCheckpoint, Llama2CTokenizer  # noqa: E402
from budgetcuda.stories_sparsity_accuracy import (  # noqa: E402
    SparsityAccuracySpec,
    evaluate_sparsity_accuracy,
    write_sparsity_accuracy_report,
)


def _policies(value: str) -> list[SparsityAccuracySpec]:
    result = []
    try:
        for item in value.split(","):
            threshold, fraction = item.strip().split(":", 1)
            result.append(SparsityAccuracySpec(int(threshold), float(fraction)))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "policies must look like 1:0.5,2:0.5,4:1.0"
        ) from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one policy is required")
    return result


def _stories(path: Path, limit: int) -> list[str]:
    text = path.read_text(encoding="utf-8")
    candidates = text.split("<|endoftext|>") if "<|endoftext|>" in text else text.split("\n\n")
    return [item.strip() for item in candidates if item.strip()][:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--stories", type=int, default=32)
    parser.add_argument("--tokens-per-story", type=int, default=64)
    parser.add_argument(
        "--policies", type=_policies,
        default=_policies("1:0.5,2:0.5,3:0.5,4:1.0,4:0.5"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = Llama2CCheckpoint(args.checkpoint, require_stories15m=True)
    baseline = (
        Llama2CCheckpoint(args.baseline_checkpoint, require_stories15m=True)
        if args.baseline_checkpoint else None
    )
    tokenizer = Llama2CTokenizer(args.tokenizer, checkpoint.config.vocab_size)
    texts = _stories(args.corpus, args.stories)
    sequences = [
        tokenizer.encode(text, bos=True, eos=False)[:args.tokens_per_story]
        for text in texts
    ]
    report = evaluate_sparsity_accuracy(
        checkpoint, sequences, args.policies, baseline_checkpoint=baseline,
    )
    report["evaluation_texts"] = texts
    json_path, markdown_path = write_sparsity_accuracy_report(report, args.output)
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(markdown_path),
        "fp32_perplexity": report["fp32_perplexity"],
        "dense_w4a8_perplexity": report["dense_w4a8"]["perplexity"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
