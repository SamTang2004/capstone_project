"""Fine-tune DeiT-Tiny, then run matched W8A8 and W4A4 QAT experiments."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from quantization import prepare_qat_model, quantizer_count, set_observers


CLASS_NAMES = (
    "apple_fresh",
    "apple_rotten",
    "banana_fresh",
    "banana_rotten",
    "orange_fresh",
    "orange_rotten",
)
MODEL_NAME = "deit_tiny_patch16_224.fb_in1k"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ManifestDataset(Dataset):
    def __init__(self, manifest: Path, image_root: Path, split: str, transform) -> None:
        self.image_root = image_root
        self.transform = transform
        with manifest.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.samples = [
            (image_root / row["relative_path"], int(row["class_index"]))
            for row in rows
            if row["split"] == split
        ]
        if not self.samples:
            raise RuntimeError(f"No {split} samples found in {manifest}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, label


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=here / "splits" / "manifest.csv")
    parser.add_argument("--image-root", type=Path, default=here / "data" / "cache_256")
    parser.add_argument("--output-dir", type=Path, default=here / "artifacts")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=here / "data" / "pretrained" / "deit_tiny_patch16_224-a1311bcf.pth",
    )
    parser.add_argument("--float-epochs", type=int, default=20)
    parser.add_argument("--qat-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--float-lr", type=float, default=3e-5)
    parser.add_argument("--qat-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_loaders(args: argparse.Namespace, seed: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.70, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(12, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(248, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    datasets = (
        ManifestDataset(args.manifest, args.image_root, "train", train_transform),
        ManifestDataset(args.manifest, args.image_root, "validation", evaluation_transform),
        ManifestDataset(args.manifest, args.image_root, "test", evaluation_transform),
    )
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": args.device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "worker_init_fn": seed_worker,
    }
    return (
        DataLoader(datasets[0], shuffle=True, generator=generator, **common),
        DataLoader(datasets[1], shuffle=False, **common),
        DataLoader(datasets[2], shuffle=False, **common),
    )


def class_weights(loader: DataLoader, device: torch.device) -> torch.Tensor:
    counts = Counter(label for _, label in loader.dataset.samples)
    weights = torch.tensor([1.0 / counts[index] for index in range(len(CLASS_NAMES))], device=device)
    return weights / weights.sum() * len(CLASS_NAMES)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> float:
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(loader, desc="train", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * labels.size(0)
    return running_loss / len(loader.dataset)


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(inputs)
            loss = criterion(logits, labels)
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.cpu().tolist())
        total_loss += loss.item() * labels.size(0)
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(CLASS_NAMES))))
    per_class_accuracy = {
        name: float(matrix[index, index] / matrix[index].sum())
        for index, name in enumerate(CLASS_NAMES)
    }
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(np.mean(np.asarray(predictions) == np.asarray(targets))),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": matrix.tolist(),
        "classification_report": classification_report(
            targets,
            predictions,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            digits=4,
            zero_division=0,
        ),
    }


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_checkpoint(
    output: Path,
    model: nn.Module,
    stage: str,
    metrics: dict,
    args: argparse.Namespace,
    bits: int | None = None,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": args.model_name,
            "num_classes": len(CLASS_NAMES),
            "class_names": CLASS_NAMES,
            "stage": stage,
            "bits": bits,
            "state_dict": cpu_state_dict(model),
            "metrics": metrics,
            "quantization": None
            if bits is None
            else {
                "weights": f"signed INT{bits}, symmetric, per-output-channel",
                "activations": f"signed INT{bits}, symmetric, per-tensor",
                "attention_probabilities": f"unsigned INT{bits}, per-tensor",
                "accumulation": "floating-point simulation of INT32 accumulation",
                "float_operations": ["LayerNorm", "GELU", "softmax", "residual addition"],
            },
            "seed": args.seed,
        },
        output,
    )


def train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    observer_freeze_epochs: int = 0,
) -> tuple[nn.Module, list[dict], dict]:
    model = model.to(device)
    weights = class_weights(train_loader, device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    best_accuracy = -math.inf
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict] = []
    best_metrics: dict = {}
    for epoch in range(1, epochs + 1):
        if observer_freeze_epochs and epoch > epochs - observer_freeze_epochs:
            set_observers(model, False)
        started = time.perf_counter()
        loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler)
        metrics = evaluate(model, validation_loader, device)
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            "train_loss": loss,
            "validation_loss": metrics["loss"],
            "validation_accuracy": metrics["accuracy"],
            "validation_macro_f1": metrics["macro_f1"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "seconds": elapsed,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d}/{epochs} train_loss={loss:.4f} "
            f"val_acc={metrics['accuracy']:.4f} val_f1={metrics['macro_f1']:.4f} "
            f"seconds={elapsed:.1f}"
        )
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_state = cpu_state_dict(model)
            best_metrics = metrics
        scheduler.step()
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_metrics


def packed_size_estimate(model: nn.Module, bits: int) -> dict[str, float | int]:
    quantized_parameters = 0
    float_parameters = 0
    for name, parameter in model.named_parameters():
        if name.endswith("weight") and parameter.ndim >= 2:
            quantized_parameters += parameter.numel()
        else:
            float_parameters += parameter.numel()
    bytes_estimate = quantized_parameters * bits / 8 + float_parameters * 4
    return {
        "quantized_parameters": quantized_parameters,
        "float_parameters": float_parameters,
        "packed_bytes_estimate": round(bytes_estimate),
        "packed_mib_estimate": bytes_estimate / 1024**2,
    }


def build_pretrained_model(checkpoint_path: Path, model_name: str = MODEL_NAME) -> nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Official pretrained checkpoint not found: {checkpoint_path}."
        )
    model = timm.create_model(model_name, pretrained=False, num_classes=1000)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Pretrained checkpoint mismatch: {incompatible}")
    model.reset_classifier(len(CLASS_NAMES))
    return model


def main() -> None:
    args = parse_args()
    if min(args.float_epochs, args.qat_epochs, args.batch_size) <= 0 or args.workers < 0:
        raise ValueError("Epochs and batch size must be positive; workers cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        args.device if args.device != "auto" else "cpu"
    )
    args.device = device
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)

    print(f"device={device} model={args.model_name} classes={len(CLASS_NAMES)}")
    float_train, validation_loader, test_loader = build_loaders(args, args.seed)
    float_model = build_pretrained_model(args.pretrained_checkpoint, args.model_name)
    print(f"parameters={sum(parameter.numel() for parameter in float_model.parameters())}")
    print("stage=float_finetuning")
    float_model, float_history, float_validation = train_stage(
        float_model,
        float_train,
        validation_loader,
        device,
        epochs=args.float_epochs,
        learning_rate=args.float_lr,
        weight_decay=args.weight_decay,
    )
    float_test = evaluate(float_model, test_loader, device)
    save_checkpoint(
        args.output_dir / "float" / "best.pt",
        float_model,
        "float",
        {"validation": float_validation, "test": float_test},
        args,
    )

    float_cpu = copy.deepcopy(float_model).cpu()
    del float_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    comparison = {
        "experiment": "DeiT-Tiny six-class fruit freshness quantization comparison",
        "model": args.model_name,
        "classes": list(CLASS_NAMES),
        "seed": args.seed,
        "float": {"validation": float_validation, "test": float_test},
        "quantized": {},
    }
    histories = {"float": float_history}

    for bits in (8, 4):
        print(f"stage=qat_int{bits}")
        seed_everything(args.seed)
        train_loader, validation_loader, test_loader = build_loaders(args, args.seed)
        quantized_model = prepare_qat_model(float_cpu, bits)
        print(f"fake_quantizers={quantizer_count(quantized_model)}")
        quantized_model, history, validation_metrics = train_stage(
            quantized_model,
            train_loader,
            validation_loader,
            device,
            epochs=args.qat_epochs,
            learning_rate=args.qat_lr,
            weight_decay=args.weight_decay,
            observer_freeze_epochs=min(2, args.qat_epochs // 3),
        )
        test_metrics = evaluate(quantized_model, test_loader, device)
        size = packed_size_estimate(quantized_model, bits)
        stage_name = f"int{bits}"
        comparison["quantized"][stage_name] = {
            "validation": validation_metrics,
            "test": test_metrics,
            "size": size,
        }
        histories[stage_name] = history
        save_checkpoint(
            args.output_dir / stage_name / "best_fake_quant.pt",
            quantized_model,
            stage_name,
            comparison["quantized"][stage_name],
            args,
            bits,
        )
        del quantized_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    int8_accuracy = comparison["quantized"]["int8"]["test"]["accuracy"]
    int4_accuracy = comparison["quantized"]["int4"]["test"]["accuracy"]
    comparison["accuracy_delta_int4_minus_int8"] = int4_accuracy - int8_accuracy
    comparison["configuration"] = {
        "float_epochs": args.float_epochs,
        "qat_epochs": args.qat_epochs,
        "batch_size": args.batch_size,
        "float_lr": args.float_lr,
        "qat_lr": args.qat_lr,
        "weight_decay": args.weight_decay,
        "device": str(device),
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(histories, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "float_test_accuracy": float_test["accuracy"],
        "int8_test_accuracy": int8_accuracy,
        "int4_test_accuracy": int4_accuracy,
        "int4_minus_int8": int4_accuracy - int8_accuracy,
    }, indent=2))
    print(f"Saved results under {args.output_dir}")


if __name__ == "__main__":
    main()
