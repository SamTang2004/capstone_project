from pathlib import Path
import argparse
import json
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from paths import FLOAT_MODEL_DIR, TRAINING_DATA_DIR


CLASS_RULES = {
    "apple": ["apple"],
    "banana": ["banana"],
    "orange": ["orange"],
}


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
C1_OUT = 12
C2_OUT = 24
C3_OUT = 48


class GroupedFruitDataset(Dataset):
    """
    Reads images from:
        Training/<case-folder>/*.jpg

    Maps folders to 3 fruit-type labels:
        Apple ...  -> apple
        Banana ... -> banana
        Orange ... -> orange
    """

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = ["apple", "banana", "orange"]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        self.samples = []

        for folder in sorted(self.root_dir.iterdir()):
            if not folder.is_dir():
                continue

            folder_lower = folder.name.lower()

            label_name = None
            for class_name, prefixes in CLASS_RULES.items():
                if any(folder_lower.startswith(prefix) for prefix in prefixes):
                    label_name = class_name
                    break

            if label_name is None:
                continue

            label_idx = self.class_to_idx[label_name]

            for image_path in sorted(folder.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, label_idx, folder.name))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No apple/banana/orange images found under {self.root_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_idx, folder_name = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx


class SmallFruitCNN(nn.Module):
    """
    Small FPGA-friendly CNN.

    Input:
        3 x 100 x 100 or 3 x 64 x 64

    Output:
        3 logits
    """

    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, C1_OUT, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C1_OUT),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(C1_OUT, C2_OUT, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C2_OUT),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(C2_OUT, C3_OUT, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(C3_OUT),
            nn.ReLU(inplace=True),

            # Avoid large fully connected layers.
            # This is better for later FPGA deployment.
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Linear(C3_OUT, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def stratified_split(dataset, val_fraction=0.2, seed=42):
    """
    Create train/validation split from a single Training directory,
    preserving class balance.
    """

    random.seed(seed)

    indices_by_class = {}

    for idx, (_, label_idx, _) in enumerate(dataset.samples):
        indices_by_class.setdefault(label_idx, []).append(idx)

    train_indices = []
    val_indices = []

    for label_idx, indices in indices_by_class.items():
        random.shuffle(indices)

        n_val = max(1, int(len(indices) * val_fraction))

        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    return train_indices, val_indices


def count_labels(dataset, indices):
    counts = Counter()

    for idx in indices:
        _, label_idx, _ = dataset.samples[idx]
        class_name = dataset.class_names[label_idx]
        counts[class_name] += 1

    return counts


@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()

    y_true = []
    y_pred = []

    total = 0
    correct = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.numel()

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    accuracy = correct / total

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred)

    return accuracy, report, cm


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, labels in tqdm(loader, desc="training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=Path, default=TRAINING_DATA_DIR)
    parser.add_argument("--image-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=FLOAT_MODEL_DIR)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),

        # Mild augmentations for camera/conveyor robustness.
        transforms.RandomRotation(12),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.06, 0.06),
            scale=(0.90, 1.10),
        ),
        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.15,
        ),

        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    # Build one dataset without transform first so we can split.
    base_dataset = GroupedFruitDataset(args.data_dir, transform=None)

    train_indices, val_indices = stratified_split(
        base_dataset,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )

    # Separate dataset objects so train and val can use different transforms.
    train_dataset_full = GroupedFruitDataset(args.data_dir, transform=train_transform)
    val_dataset_full = GroupedFruitDataset(args.data_dir, transform=val_transform)

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    print("Classes:")
    for idx, name in enumerate(base_dataset.class_names):
        print(f"  {idx}: {name}")

    print()
    print(f"Total selected images: {len(base_dataset)}")
    print(f"Training images:       {len(train_dataset)}")
    print(f"Validation images:     {len(val_dataset)}")

    print()
    print("Training class counts:")
    print(count_labels(base_dataset, train_indices))

    print()
    print("Validation class counts:")
    print(count_labels(base_dataset, val_indices))
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    model = SmallFruitCNN(num_classes=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_acc, report, cm = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            class_names=base_dataset.class_names,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"loss={train_loss:.5f} | "
            f"val_acc={val_acc:.5f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": base_dataset.class_names,
                "image_size": args.image_size,
                "val_accuracy": best_val_acc,
                "architecture": "SmallFruitCNN",
            }

            torch.save(checkpoint, out_dir / "best.pt")

            with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
                f.write(f"best_val_accuracy={best_val_acc:.6f}\n\n")
                f.write(report)
                f.write("\nConfusion matrix:\n")
                f.write(str(cm))
                f.write("\n")

            with open(out_dir / "config.json", "w", encoding="utf-8") as f:
                config = vars(args).copy()
                config["data_dir"] = str(args.data_dir)
                config["out_dir"] = str(args.out_dir)
                json.dump(config, f, indent=2)

    print()
    print(f"Best validation accuracy: {best_val_acc:.5f}")
    print(f"Saved best checkpoint to: {out_dir / 'best.pt'}")
    print(f"Saved metrics to:         {out_dir / 'metrics.txt'}")


if __name__ == "__main__":
    main()
