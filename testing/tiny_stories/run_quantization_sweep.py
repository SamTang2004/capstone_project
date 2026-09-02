"""Measure TinyStories WXAY fake-quant sensitivity against the FP32 reference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "FPGA"))

from budgetcuda.stories import Llama2CCheckpoint, Llama2CTokenizer  # noqa: E402
from budgetcuda.stories_quantization import (  # noqa: E402
    StoriesQuantizationSpec,
    evaluate_quantization_sweep,
    write_quantization_report,
)


DEFAULT_STORIES = (
    "Once upon a time, there was a little girl named Lily. She loved to play outside in the park.",
    "Tom had a small red ball. One day, he took it to school and shared it with his friend.",
    "The friendly dog was hungry, so Mia gave him some food and a bowl of clean water.",
    "A little bird wanted to fly above the tall tree, but first it had to practice every day.",
)


def _bits(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("bits must be comma-separated integers") from exc
    if not result or any(bit < 2 or bit > 16 for bit in result):
        raise argparse.ArgumentTypeError("every bit width must be in 2..16")
    return result


def _configs(value: str) -> list[tuple[int, int]]:
    result = []
    for item in value.split(","):
        label = item.strip().upper()
        if not label:
            continue
        if not label.startswith("W") or "A" not in label[1:]:
            raise argparse.ArgumentTypeError("configs must look like W8A8,W6A6")
        weight, activation = label[1:].split("A", 1)
        try:
            pair = (int(weight), int(activation))
        except ValueError as exc:
            raise argparse.ArgumentTypeError("configs must look like W8A8,W6A6") from exc
        if any(bit < 2 or bit > 16 for bit in pair):
            raise argparse.ArgumentTypeError("every bit width must be in 2..16")
        result.append(pair)
    if not result:
        raise argparse.ArgumentTypeError("at least one WXAY config is required")
    return result


def _stories(path: Path | None, limit: int) -> list[str]:
    if path is None:
        return list(DEFAULT_STORIES[:limit])
    text = path.read_text(encoding="utf-8")
    if "<|endoftext|>" in text:
        candidates = text.split("<|endoftext|>")
    else:
        candidates = text.split("\n\n")
    return [item.strip() for item in candidates if item.strip()][:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument(
        "--baseline-checkpoint", type=Path,
        help="optional original FP32 teacher used to judge a fine-tuned/QAT checkpoint",
    )
    parser.add_argument("--corpus", type=Path, help="TinyStories validation text")
    parser.add_argument("--stories", type=int, default=4)
    parser.add_argument("--tokens-per-story", type=int, default=32)
    parser.add_argument("--weight-bits", type=_bits, default=_bits("8,6,4,3,2"))
    parser.add_argument("--activation-bits", type=_bits, default=_bits("8,6,4,3,2"))
    parser.add_argument(
        "--configs", type=_configs,
        help="explicit comma-separated pairs such as W8A8,W6A6; overrides the bit grids",
    )
    parser.add_argument("--weight-granularity", choices=("per_tensor", "per_output"), default="per_tensor")
    parser.add_argument("--sample-prompt", default="Once upon a time")
    parser.add_argument("--sample-new-tokens", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = Llama2CCheckpoint(args.checkpoint, require_stories15m=True)
    baseline_checkpoint = (
        Llama2CCheckpoint(args.baseline_checkpoint, require_stories15m=True)
        if args.baseline_checkpoint else None
    )
    tokenizer = Llama2CTokenizer(args.tokenizer, checkpoint.config.vocab_size)
    texts = _stories(args.corpus, args.stories)
    sequences = [
        tokenizer.encode(text, bos=True, eos=False)[:args.tokens_per_story]
        for text in texts
    ]
    bit_pairs = args.configs or [
        (weight, activation)
        for weight in args.weight_bits for activation in args.activation_bits
    ]
    specs = [StoriesQuantizationSpec(weight, activation, args.weight_granularity)
             for weight, activation in bit_pairs]
    prompt = tokenizer.encode(args.sample_prompt, bos=True, eos=False)
    report = evaluate_quantization_sweep(
        checkpoint, sequences, specs, baseline_checkpoint=baseline_checkpoint,
        sample_prompt_tokens=prompt,
        sample_new_tokens=args.sample_new_tokens,
    )
    for item in report["evaluations"]:
        if item.get("greedy_tokens"):
            item["greedy_text"] = tokenizer.decode(item["greedy_tokens"])
    report["baseline_greedy_text"] = tokenizer.decode(report["baseline_greedy_tokens"])
    report["evaluation_texts"] = texts
    json_path, markdown_path = write_quantization_report(report, args.output)
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(markdown_path),
        "baseline_perplexity": report["baseline_perplexity"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
