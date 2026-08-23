"""Run the matched scaled-E2M1 FP4 QAT experiment from the saved float model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import timm
import torch

from quantization import prepare_fp4_qat_model, quantizer_count
from train_deit_tiny import (
    CLASS_NAMES,
    MODEL_NAME,
    build_loaders,
    cpu_state_dict,
    evaluate,
    packed_size_estimate,
    seed_everything,
    train_stage,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=here / "splits" / "manifest.csv")
    parser.add_argument("--image-root", type=Path, default=here / "data" / "cache_256")
    parser.add_argument("--float-checkpoint", type=Path, default=here / "artifacts" / "float" / "best.pt")
    parser.add_argument("--output-dir", type=Path, default=here / "artifacts")
    parser.add_argument("--qat-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--qat-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        args.device if args.device != "auto" else "cpu"
    )
    args.device = device
    seed_everything(args.seed)
    checkpoint = torch.load(args.float_checkpoint, map_location="cpu", weights_only=True)
    model_name = checkpoint.get("model_name", MODEL_NAME)
    float_model = timm.create_model(model_name, pretrained=False, num_classes=len(CLASS_NAMES))
    float_model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = prepare_fp4_qat_model(float_model)
    print(f"device={device} stage=qat_fp4_e2m1 fake_quantizers={quantizer_count(model)}")

    train_loader, validation_loader, test_loader = build_loaders(args, args.seed)
    model, history, validation_metrics = train_stage(
        model,
        train_loader,
        validation_loader,
        device,
        epochs=args.qat_epochs,
        learning_rate=args.qat_lr,
        weight_decay=args.weight_decay,
        observer_freeze_epochs=min(2, args.qat_epochs // 3),
    )
    test_metrics = evaluate(model, test_loader, device)
    size = packed_size_estimate(model, 4)
    output = args.output_dir / "fp4_e2m1" / "best_fake_quant.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "num_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "stage": "fp4_e2m1",
            "bits": 4,
            "quantization_format": "fp4_e2m1",
            "state_dict": cpu_state_dict(model),
            "metrics": {"validation": validation_metrics, "test": test_metrics, "size": size},
            "quantization": {
                "elements": "scaled FP4 E2M1 codebook",
                "weights": "per-output-channel scale",
                "activations": "per-tensor scale",
                "attention_probabilities": "nonnegative E2M1 values with per-tensor scale",
                "float_operations": ["LayerNorm", "GELU", "softmax", "residual addition"],
                "not_nvfp4": True,
            },
            "seed": args.seed,
        },
        output,
    )
    result = {
        "format": "scaled FP4 E2M1",
        "validation": validation_metrics,
        "test": test_metrics,
        "size": size,
        "history": history,
        "configuration": {
            "qat_epochs": args.qat_epochs,
            "batch_size": args.batch_size,
            "qat_lr": args.qat_lr,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "device": str(device),
        },
    }
    (args.output_dir / "fp4_e2m1" / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "validation_accuracy": validation_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
    }, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
