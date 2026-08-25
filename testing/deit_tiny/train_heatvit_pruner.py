"""Train and export HeatViT token selectors for the W4A8 fruit DeiT model."""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from export_budgetcuda import export_linear, quantize_symmetric
from heatvit_model import HeatViTDeiT
from quantization import prepare_qat_model, set_observers
from train_deit_tiny import CLASS_NAMES, build_loaders, cpu_state_dict, evaluate, seed_everything


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=here / "artifacts" / "w4a8_per_tensor" / "best_fake_quant.pt")
    parser.add_argument("--base-export", type=Path, default=here / "artifacts" / "budgetcuda_w4a8")
    parser.add_argument("--output-dir", type=Path, default=here / "artifacts" / "heatvit_w4a8")
    parser.add_argument("--manifest", type=Path, default=here / "splits" / "manifest.csv")
    parser.add_argument("--image-root", type=Path, default=here / "data" / "cache_256")
    parser.add_argument("--selector-blocks", type=int, nargs="+", default=(2, 5, 8))
    parser.add_argument("--output-tokens", type=int, nargs="+", default=(151, 106, 44))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--distill-weight", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--selector-bits", type=int, choices=(4, 8), default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def load_w4a8(checkpoint_path: Path) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint.get("weight_bits", checkpoint.get("bits", 0))) != 4 or int(checkpoint.get("activation_bits", 8)) != 8:
        raise ValueError("HeatViT training requires the W4A8 checkpoint")
    if tuple(checkpoint.get("class_names", ())) != tuple(CLASS_NAMES):
        raise ValueError("Checkpoint class order does not match the fruit dataset")
    model = timm.create_model(checkpoint["model_name"], pretrained=False, num_classes=len(CLASS_NAMES))
    model = prepare_qat_model(model, 4, 8, weight_per_channel=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    set_observers(model, False)
    return model


@torch.inference_mode()
def evaluate_with_trace(model: HeatViTDeiT, loader, device: torch.device) -> dict:
    metrics = evaluate(model, loader, device)
    trace_counts = None
    for inputs, _ in loader:
        _, trace = model(inputs[:1].to(device), return_trace=True)
        trace_counts = [{"after_block": item.after_block, "input_tokens": item.input_tokens,
                         "output_tokens": item.output_tokens} for item in trace]
        break
    metrics["selector_trace"] = trace_counts or []
    return metrics


def train(model: HeatViTDeiT, teacher: nn.Module, train_loader, validation_loader,
          device: torch.device, args: argparse.Namespace):
    model.to(device); teacher.to(device).eval()
    optimizer = torch.optim.AdamW(model.selector_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_accuracy, best_state, best_metrics = -math.inf, None, None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); model.backbone.eval()
        running_loss = 0.0; started = time.perf_counter()
        for inputs, labels in tqdm(train_loader, desc=f"selector {epoch}/{args.epochs}", leave=False):
            inputs = inputs.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                teacher_logits = teacher(inputs)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(inputs)
                label_loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
                temperature = args.temperature
                distill = F.kl_div(F.log_softmax(logits / temperature, dim=1),
                                   F.softmax(teacher_logits / temperature, dim=1), reduction="batchmean") * temperature**2
                loss = (1.0 - args.distill_weight) * label_loss + args.distill_weight * distill
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.selectors.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            running_loss += float(loss.item()) * labels.shape[0]
        metrics = evaluate_with_trace(model, validation_loader, device)
        row = {"epoch": epoch, "train_loss": running_loss / len(train_loader.dataset),
               "validation_accuracy": metrics["accuracy"], "validation_macro_f1": metrics["macro_f1"],
               "seconds": time.perf_counter() - started, "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row); print(json.dumps(row))
        if metrics["accuracy"] > best_accuracy:
            best_accuracy, best_metrics = metrics["accuracy"], metrics
            best_state = cpu_state_dict(model.selectors)
        scheduler.step()
    if best_state is None: raise RuntimeError("Selector training produced no checkpoint")
    model.selectors.load_state_dict(best_state)
    return history, best_metrics


def hardware_quantized_selectors(model: HeatViTDeiT, bits: int) -> HeatViTDeiT:
    """Clone selectors with the exact symmetric weight quantization used by export."""
    quantized = copy.deepcopy(model)
    with torch.no_grad():
        for selector in quantized.selectors:
            values, scale = quantize_symmetric(selector.score.weight, bits)
            selector.score.weight.copy_(values.to(selector.score.weight.dtype) * scale)
    return quantized


def export_heatvit(model: HeatViTDeiT, args: argparse.Namespace, validation: dict, test: dict,
                   hardware_validation: dict, hardware_test: dict, history: list[dict]) -> Path:
    output = args.output_dir
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(args.base_export, output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selector_entries = []
    selector_state = model.selectors.state_dict()
    export_state = {}
    for index, (block, output_tokens) in enumerate(zip(args.selector_blocks, args.output_tokens)):
        source = f"{index}.score"
        target = f"heatvit.selectors.{index}.score"
        export_state[target + ".weight"] = selector_state[source + ".weight"]
        export_state[target + ".bias"] = selector_state[source + ".bias"]
        export_linear(export_state, target, output, 1.0, 1.0, manifest, args.selector_bits)
        manifest["tensors"][target]["weight_bits"] = args.selector_bits
        selector_entries.append({"after_block": int(block), "output_tokens": int(output_tokens),
                                 "scorer": target, "package_token": True})
    manifest["heatvit"] = {"version": 1, "enabled_by_default": False,
                           "selector_bits": args.selector_bits, "selectors": selector_entries,
                           "training": {"checkpoint": str(args.checkpoint), "validation": validation,
                                        "test": test, "hardware_quantized_validation": hardware_validation,
                                        "hardware_quantized_test": hardware_test}}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checkpoint = output / "selectors.pt"
    torch.save({"selector_blocks": tuple(args.selector_blocks), "output_tokens": tuple(args.output_tokens),
                "state_dict": selector_state, "validation": validation, "test": test}, checkpoint)
    (output / "training_results.json").write_text(json.dumps({"history": history, "validation": validation,
                                                               "test": test,
                                                               "hardware_quantized_validation": hardware_validation,
                                                               "hardware_quantized_test": hardware_test}, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    args = parse_args()
    if len(args.selector_blocks) != len(args.output_tokens) or args.epochs <= 0:
        raise ValueError("Selector configuration and epoch count are invalid")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          args.device if args.device != "auto" else "cpu")
    args.device = device
    seed_everything(args.seed)
    backbone = load_w4a8(args.checkpoint)
    teacher = copy.deepcopy(backbone)
    for parameter in backbone.parameters(): parameter.requires_grad_(False)
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    model = HeatViTDeiT(backbone, tuple(args.selector_blocks), tuple(args.output_tokens), args.selector_bits)
    train_loader, validation_loader, test_loader = build_loaders(args, args.seed)
    print(f"device={device} selectors={list(zip(args.selector_blocks, args.output_tokens))} train={len(train_loader.dataset)}")
    history, validation = train(model, teacher, train_loader, validation_loader, device, args)
    test = evaluate_with_trace(model, test_loader, device)
    quantized_model = hardware_quantized_selectors(model, args.selector_bits).to(device)
    hardware_validation = evaluate_with_trace(quantized_model, validation_loader, device)
    hardware_test = evaluate_with_trace(quantized_model, test_loader, device)
    manifest = export_heatvit(model.cpu(), args, validation, test, hardware_validation, hardware_test, history)
    print(json.dumps({"validation_accuracy": validation["accuracy"], "validation_macro_f1": validation["macro_f1"],
                      "test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"],
                      "hardware_quantized_test_accuracy": hardware_test["accuracy"],
                      "manifest": str(manifest)}, indent=2))


if __name__ == "__main__":
    main()
