"""Reload and evaluate a saved DeiT-Tiny float or fake-quantized checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import timm
import torch

from quantization import prepare_fp4_qat_model, prepare_qat_model, set_observers
from train_deit_tiny import CLASS_NAMES, MODEL_NAME, build_loaders, evaluate, seed_everything


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--manifest", type=Path, default=here / "splits" / "manifest.csv")
    parser.add_argument("--image-root", type=Path, default=here / "data" / "cache_256")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        args.device if args.device != "auto" else "cpu"
    )
    args.device = device
    seed_everything(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if tuple(checkpoint["class_names"]) != CLASS_NAMES:
        raise RuntimeError("Checkpoint class order does not match this experiment")
    model_name = checkpoint.get("model_name", MODEL_NAME)
    model = timm.create_model(model_name, pretrained=False, num_classes=len(CLASS_NAMES))
    bits = checkpoint.get("weight_bits", checkpoint.get("bits"))
    activation_bits = checkpoint.get("activation_bits", bits)
    if checkpoint.get("quantization_format") == "fp4_e2m1":
        model = prepare_fp4_qat_model(model)
        set_observers(model, False)
    elif bits is not None:
        model = prepare_qat_model(model, bits, activation_bits)
        set_observers(model, False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    _, _, test_loader = build_loaders(args, args.seed)
    metrics = evaluate(model, test_loader, device)
    print(json.dumps({
        "checkpoint": str(args.checkpoint),
        "stage": checkpoint["stage"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_accuracy": metrics["per_class_accuracy"],
        "confusion_matrix": metrics["confusion_matrix"],
    }, indent=2))


if __name__ == "__main__":
    main()
