"""Fine-tune the fruit classifier with BudgetCUDA signed INT4 fake quantization."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from paths import FLOAT_CHECKPOINT, INT4_MODEL_DIR, TRAINING_DATA_DIR
from qat_model import (
    CLASS_NAMES,
    GroupedFruitDataset,
    QATFruitCNN64,
    count_labels,
    evaluate,
    load_float_checkpoint_into_qat,
    prepare_budgetcuda_qat,
    stratified_split,
)


def _train_epoch(model, loader, optimizer, criterion, device, max_batches):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * labels.numel()
        total_samples += labels.numel()
    if total_samples == 0:
        raise RuntimeError("No training batches were processed")
    return total_loss / total_samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=TRAINING_DATA_DIR)
    parser.add_argument("--float-checkpoint", type=Path, default=FLOAT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=INT4_MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument(
        "--channels",
        type=int,
        nargs=3,
        metavar=("C1", "C2", "C3"),
        default=(16, 32, 64),
        help="Convolution widths; the INT4 production default is 16 32 64.",
    )
    args = parser.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        parser.error("--epochs and --batch-size must be positive")
    if args.max_train_batches is not None and args.max_train_batches <= 0:
        parser.error("--max-train-batches must be positive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    train_transform = transforms.Compose(
        [
            transforms.Resize((64, 64), antialias=True),
            transforms.RandomRotation(12),
            transforms.RandomAffine(0, translate=(0.06, 0.06), scale=(0.90, 1.10)),
            transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15),
            transforms.ToTensor(),
        ]
    )
    val_transform = transforms.Compose(
        [transforms.Resize((64, 64), antialias=True), transforms.ToTensor()]
    )
    base = GroupedFruitDataset(args.data_dir)
    train_indices, val_indices = stratified_split(base, args.val_fraction, args.seed)
    train_set = Subset(GroupedFruitDataset(args.data_dir, train_transform), train_indices)
    val_set = Subset(GroupedFruitDataset(args.data_dir, val_transform), val_indices)
    loader_args = {"batch_size": args.batch_size, "num_workers": 0}
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    channels = tuple(args.channels)
    float_model = QATFruitCNN64(len(CLASS_NAMES), channels=channels)
    load_float_checkpoint_into_qat(float_model, args.float_checkpoint)
    model = prepare_budgetcuda_qat(float_model).to(device)

    counts = count_labels(base, train_indices)
    class_weights = torch.tensor(
        [1.0 / counts[name] for name in CLASS_NAMES], dtype=torch.float32, device=device
    )
    class_weights = class_weights / class_weights.sum() * len(CLASS_NAMES)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        loss = _train_epoch(
            model, train_loader, optimizer, criterion, device, args.max_train_batches
        )
        accuracy, report, confusion = evaluate(model, val_loader, device, CLASS_NAMES)
        print(f"epoch={epoch}/{args.epochs} loss={loss:.6f} val_accuracy={accuracy:.6f}")
        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            best_state = copy.deepcopy(model.state_dict())
            (args.output_dir / "metrics.txt").write_text(
                f"best_val_accuracy={accuracy:.6f}\n\n{report}\n"
                f"Confusion matrix:\n{confusion}\n",
                encoding="utf-8",
            )

    checkpoint_path = args.output_dir / "best_int4_qat_fakequant.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "architecture": "QATFruitCNN64",
            "class_names": CLASS_NAMES,
            "image_size": 64,
            "val_accuracy": best_accuracy,
            "quant_min": -8,
            "quant_max": 7,
            "dtype": "signed_int4_fake_quant",
            "channels": list(channels),
        },
        checkpoint_path,
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {
                "data_dir": str(args.data_dir.resolve()),
                "float_checkpoint": str(args.float_checkpoint.resolve()),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "max_train_batches": args.max_train_batches,
                "device": str(device),
                "profile": "signed INT4, per-tensor symmetric, zero point 0",
                "channels": list(channels),
                "checkpoint": str(checkpoint_path.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved={checkpoint_path}")


if __name__ == "__main__":
    main()
