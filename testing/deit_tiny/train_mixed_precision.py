"""Run DeiT mixed-precision QAT from an existing float checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import timm
import torch

from quantization import prepare_qat_model, quantizer_count
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
    parser.add_argument("--weight-bits", type=int, default=4)
    parser.add_argument("--activation-bits", type=int, default=8)
    parser.add_argument(
        "--weight-scale", choices=("per-channel", "per-tensor"),
        default="per-tensor",
        help="Use per-tensor for the current FPGA scalar-requantization contract",
    )
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
    if not 2 <= args.weight_bits <= 16 or not 2 <= args.activation_bits <= 16:
        raise ValueError("weight and activation widths must be in [2, 16]")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        args.device if args.device != "auto" else "cpu"
    )
    args.device = device
    seed_everything(args.seed)

    checkpoint = torch.load(args.float_checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("stage") != "float":
        raise ValueError("--float-checkpoint must reference a float checkpoint")
    model_name = checkpoint.get("model_name", MODEL_NAME)
    float_model = timm.create_model(model_name, pretrained=False, num_classes=len(CLASS_NAMES))
    float_model.load_state_dict(checkpoint["state_dict"], strict=True)
    weight_per_channel = args.weight_scale == "per-channel"
    model = prepare_qat_model(
        float_model, args.weight_bits, args.activation_bits,
        weight_per_channel=weight_per_channel)
    stage = f"w{args.weight_bits}a{args.activation_bits}"
    if not weight_per_channel:
        stage += "_per_tensor"
    print(f"device={device} stage=qat_{stage} fake_quantizers={quantizer_count(model)}")

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
    size = packed_size_estimate(model, args.weight_bits)
    output_dir = args.output_dir / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "best_fake_quant.pt"
    metrics = {"validation": validation_metrics, "test": test_metrics, "size": size}
    torch.save(
        {
            "model_name": model_name,
            "num_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "stage": stage,
            "bits": args.weight_bits,
            "weight_bits": args.weight_bits,
            "activation_bits": args.activation_bits,
            "state_dict": cpu_state_dict(model),
            "metrics": metrics,
            "quantization": {
                "weights": (
                    f"signed INT{args.weight_bits}, symmetric, "
                    f"{args.weight_scale}"
                ),
                "activations": f"signed INT{args.activation_bits}, symmetric, per-tensor",
                "attention_probabilities": f"unsigned INT{args.activation_bits}, per-tensor",
                "accumulation": "floating-point simulation of integer accumulation",
                "float_operations": ["LayerNorm", "GELU", "softmax", "residual addition"],
            },
            "seed": args.seed,
        },
        output,
    )
    result = {
        "format": stage.upper(),
        "validation": validation_metrics,
        "test": test_metrics,
        "size": size,
        "history": history,
        "configuration": vars(args) | {"device": str(device)},
    }
    result["configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in result["configuration"].items()
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "validation_accuracy": validation_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "packed_mib_estimate": size["packed_mib_estimate"],
    }, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
