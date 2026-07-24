"""Fine-tune the fruit classifier with the signed BudgetCUDA QAT profile."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from paths import FLOAT_CHECKPOINT, MODELS_DIR, TRAINING_DATA_DIR
from qat_model import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    LR,
    SEED,
    VAL_FRACTION,
    GroupedFruitDataset,
    QATFruitCNN64,
    count_labels,
    evaluate,
    load_float_checkpoint_into_qat,
    prepare_budgetcuda_qat,
    stratified_split,
    train_one_epoch,
)


DEFAULT_OUTPUT = MODELS_DIR / "fruit3_budgetcuda_qat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--float-checkpoint", type=Path, default=FLOAT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=TRAINING_DATA_DIR)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0 or args.workers < 0:
        raise ValueError("epochs, batch-size, and learning-rate must be positive; workers cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available")

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.90, 1.10)),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True), transforms.ToTensor()])
    base = GroupedFruitDataset(args.data_dir)
    train_indices, val_indices = stratified_split(base, VAL_FRACTION, SEED)
    train_set = Subset(GroupedFruitDataset(args.data_dir, train_transform), train_indices)
    val_set = Subset(GroupedFruitDataset(args.data_dir, eval_transform), val_indices)
    loader_args = {"num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_args)

    model = QATFruitCNN64(num_classes=len(CLASS_NAMES))
    load_float_checkpoint_into_qat(model, args.float_checkpoint)
    model = prepare_budgetcuda_qat(model).to(device)
    counts = count_labels(base, train_indices)
    class_weights = torch.tensor([1.0 / counts[name] for name in CLASS_NAMES], device=device)
    class_weights = class_weights / class_weights.sum() * len(CLASS_NAMES)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_accuracy, report, matrix = evaluate(model, val_loader, device, CLASS_NAMES)
    best_state = copy.deepcopy(model.state_dict())
    print(f"BudgetCUDA QAT profile: signed qint8, per-tensor symmetric, zero point 0")
    print(f"Initial validation accuracy: {best_accuracy:.5f}")
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        accuracy, report, matrix = evaluate(model, val_loader, device, CLASS_NAMES)
        print(f"Epoch {epoch:03d}/{args.epochs} | loss={loss:.5f} | val_acc={accuracy:.5f}")
        if accuracy >= best_accuracy:
            best_accuracy, best_state = accuracy, copy.deepcopy(model.state_dict())

    checkpoint = {
        "model_state_dict": best_state,
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "val_accuracy": best_accuracy,
        "architecture": "QATFruitCNN64_BudgetCUDA",
        "quantization": "signed qint8, per-tensor symmetric, zero point 0",
    }
    torch.save(checkpoint, args.output_dir / "best_budgetcuda_qat_fakequant.pt")
    (args.output_dir / "metrics.txt").write_text(f"best_val_accuracy={best_accuracy:.6f}\n\n{report}\nConfusion matrix:\n{matrix}\n", encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps({
        "data_dir": str(args.data_dir), "float_checkpoint": str(args.float_checkpoint), "epochs": args.epochs,
        "batch_size": args.batch_size, "learning_rate": args.learning_rate, "device": str(device),
        "profile": "signed qint8, per-tensor symmetric, zero point 0",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output_dir / 'best_budgetcuda_qat_fakequant.pt'}")


if __name__ == "__main__":
    main()
