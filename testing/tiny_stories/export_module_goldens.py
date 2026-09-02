"""Export independently hashed W4A8 TinyStories module golden archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "FPGA"))

from budgetcuda.stories import Llama2CCheckpoint, Llama2CTokenizer  # noqa: E402
from budgetcuda.stories_module_goldens import export_w4a8_module_goldens  # noqa: E402


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
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt")
    source.add_argument("--tokens", type=_token_ids)
    parser.add_argument(
        "--capture-position", type=int,
        help="position to capture after filling earlier K/V cache entries; defaults to the last token",
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--allow-compatible-model", action="store_true")
    args = parser.parse_args()

    checkpoint = Llama2CCheckpoint(
        args.checkpoint, require_stories15m=not args.allow_compatible_model,
    )
    tokenizer = None
    if args.tokenizer:
        tokenizer = Llama2CTokenizer(args.tokenizer, checkpoint.config.vocab_size)
    if args.prompt is not None:
        if tokenizer is None:
            parser.error("--prompt requires --tokenizer")
        tokens = tokenizer.encode(args.prompt, bos=True, eos=False)
    else:
        tokens = args.tokens
    manifest = export_w4a8_module_goldens(
        checkpoint, args.output_directory, tokens,
        capture_position=args.capture_position, tokenizer=tokenizer,
    )
    print(json.dumps({
        "manifest": str(manifest),
        "tokens": tokens,
        "capture_position": args.capture_position if args.capture_position is not None else len(tokens) - 1,
    }, indent=2))


if __name__ == "__main__":
    main()
