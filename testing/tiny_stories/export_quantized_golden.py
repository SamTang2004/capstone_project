"""Export TinyStories WXAY fake-quant logits and per-layer golden tensors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "FPGA"))

from budgetcuda.stories import Llama2CCheckpoint, Llama2CTokenizer  # noqa: E402
from budgetcuda.stories_quantization import (  # noqa: E402
    StoriesQuantizationSpec,
    export_quantized_stories_golden,
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
    parser.add_argument("checkpoint", type=Path, help="QAT float-shadow llama2.c checkpoint")
    parser.add_argument("--tokenizer", type=Path, help="llama2.c tokenizer.bin")
    parser.add_argument(
        "--source-checkpoint", type=Path,
        help="optional original checkpoint recorded as QAT provenance",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="text prompt; requires --tokenizer")
    source.add_argument("--tokens", type=_token_ids, help="comma-separated input token ids")
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument(
        "--weight-granularity", choices=("per_tensor", "per_output"),
        default="per_output",
    )
    parser.add_argument("--new-tokens", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-compatible-model", action="store_true")
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

    provenance = {"kind": "QAT float-shadow checkpoint"}
    if args.source_checkpoint:
        original = Llama2CCheckpoint(
            args.source_checkpoint, require_stories15m=not args.allow_compatible_model,
        )
        if original.config != checkpoint.config:
            parser.error("--source-checkpoint config does not match the QAT checkpoint")
        provenance["source_checkpoint"] = {
            "path": str(original.path), "sha256": original.sha256,
        }
    spec = StoriesQuantizationSpec(
        args.weight_bits, args.activation_bits, args.weight_granularity,
    )
    json_path, npz_path = export_quantized_stories_golden(
        checkpoint, args.output, tokens, spec,
        max_new_tokens=args.new_tokens, top_k=args.top_k,
        tokenizer=tokenizer, provenance=provenance,
    )
    print(json.dumps({
        "json": str(json_path),
        "npz": str(npz_path),
        "prompt_tokens": tokens,
        "quantization": spec.as_dict(),
    }, indent=2))


if __name__ == "__main__":
    main()
