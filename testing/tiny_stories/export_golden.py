"""Export stories15M FP32 logits and per-layer golden tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "FPGA"))

from budgetcuda.stories import (  # noqa: E402
    Llama2CCheckpoint,
    Llama2CTokenizer,
    export_stories_golden,
)


def _token_ids(value: str) -> list[int]:
    try:
        tokens = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("token ids must be comma-separated integers") from exc
    if not tokens:
        raise argparse.ArgumentTypeError("at least one token id is required")
    return tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="legacy llama2.c stories15M.bin")
    parser.add_argument("--tokenizer", type=Path, help="llama2.c tokenizer.bin")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="text prompt; requires --tokenizer")
    source.add_argument("--tokens", type=_token_ids, help="comma-separated input token ids")
    parser.add_argument("--new-tokens", type=int, default=0, help="greedy tokens to generate")
    parser.add_argument("--top-k", type=int, default=16, help="logits listed per step in JSON")
    parser.add_argument("--output", type=Path, required=True, help="output prefix without extension")
    parser.add_argument(
        "--allow-compatible-model", action="store_true",
        help="accept another legacy llama2.c shape instead of requiring stories15M",
    )
    args = parser.parse_args()

    checkpoint = Llama2CCheckpoint(
        args.checkpoint, require_stories15m=not args.allow_compatible_model,
    )
    tokenizer = None
    if args.tokenizer is not None:
        tokenizer = Llama2CTokenizer(args.tokenizer, checkpoint.config.vocab_size)
    if args.prompt is not None:
        if tokenizer is None:
            parser.error("--prompt requires --tokenizer")
        tokens = tokenizer.encode(args.prompt, bos=True, eos=False)
    else:
        tokens = args.tokens

    json_path, npz_path = export_stories_golden(
        checkpoint, args.output, tokens, max_new_tokens=args.new_tokens,
        top_k=args.top_k, tokenizer=tokenizer,
    )
    print(json.dumps({
        "json": str(json_path),
        "npz": str(npz_path),
        "prompt_tokens": tokens,
    }, indent=2))


if __name__ == "__main__":
    main()

