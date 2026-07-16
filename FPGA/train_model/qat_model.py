from pathlib import Path
import copy
import json
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

from paths import FLOAT_CHECKPOINT, QAT_MODEL_DIR, TRAINING_DATA_DIR


# ============================================================
# User settings
# ============================================================

DATA_DIR = TRAINING_DATA_DIR
OUT_DIR = QAT_MODEL_DIR

IMAGE_SIZE = 64
EPOCHS = 10
BATCH_SIZE = 64
LR = 1e-4
VAL_FRACTION = 0.2
SEED = 42

CLASS_NAMES = ["apple", "banana", "orange"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
C1_OUT = 12
C2_OUT = 24
C3_OUT = 48


# ============================================================
# Quantization backend
# ============================================================

def select_quant_backend():
    engines = torch.backends.quantized.supported_engines
    print("Supported quantized engines:", engines)

    if "x86" in engines:
        return "x86"
    if "fbgemm" in engines:
        return "fbgemm"
    if "onednn" in engines:
        return "onednn"
    if "qnnpack" in engines:
        return "qnnpack"

    raise RuntimeError(
        f"No usable quantized backend found. Supported engines: {engines}"
    )


def budgetcuda_qat_qconfig() -> tq.QConfig:
    """PyTorch eager-QAT profile matching BudgetCUDA's signed PL arithmetic.

    The deployed accelerator has one requantization scale per tensor and its
    activation units interpret zero as the real zero.  Therefore this profile
    deliberately uses signed, per-tensor symmetric fake quantization for both
    activations and weights.  It is built entirely from torch.ao primitives.
    """
    fake_quant = tq.FakeQuantize.with_args(
        observer=tq.MovingAverageMinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        quant_min=-127,
        quant_max=127,
        reduce_range=False,
    )
    return tq.QConfig(activation=fake_quant, weight=fake_quant)


def prepare_budgetcuda_qat(model: nn.Module) -> nn.Module:
    """Fuse and prepare a QAT model using the deployable signed profile."""
    model.eval()
    if hasattr(model, "fuse_model"):
        model.fuse_model()
    model.train()
    model.qconfig = budgetcuda_qat_qconfig()
    return tq.prepare_qat(model, inplace=False)


# ============================================================
# Dataset
# ============================================================

class GroupedFruitDataset(Dataset):
    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_names = CLASS_NAMES
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        for folder in sorted(self.root_dir.iterdir()):
            if not folder.is_dir():
                continue

            folder_name_lower = folder.name.lower()

            if folder_name_lower.startswith("apple"):
                label_name = "apple"
            elif folder_name_lower.startswith("banana"):
                label_name = "banana"
            elif folder_name_lower.startswith("orange"):
                label_name = "orange"
            else:
                continue

            label_idx = self.class_to_idx[label_name]

            for image_path in sorted(folder.rglob("*")):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, label_idx, folder.name))

        if not self.samples:
            raise RuntimeError(f"No apple/banana/orange images found under {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_idx, _ = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label_idx


def stratified_split(dataset, val_fraction=0.2, seed=42):
    random.seed(seed)

    indices_by_class = {}

    for idx, (_, label_idx, _) in enumerate(dataset.samples):
        indices_by_class.setdefault(label_idx, []).append(idx)

    train_indices = []
    val_indices = []

    for _, indices in indices_by_class.items():
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
        counts[dataset.class_names[label_idx]] += 1

    return counts


# ============================================================
# QAT model: fixed 64x64 input
# ============================================================

class QATFruitCNN64(nn.Module):
    """
    QAT-compatible CNN for fixed 64x64 input.

    Input:
        N x 3 x 64 x 64

    Shape flow:
        64x64
        -> MaxPool2d(2) = 32x32
        -> MaxPool2d(2) = 16x16
        -> Conv3 keeps 16x16
        -> AvgPool2d(16) = 1x1
        -> Linear(48 -> 3)

    This is better for FPGA/HLS than AdaptiveAvgPool2d.
    """

    def __init__(self, num_classes=3):
        super().__init__()

        self.quant = tq.QuantStub()

        self.conv1 = nn.Conv2d(3, C1_OUT, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(C1_OUT)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(C1_OUT, C2_OUT, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(C2_OUT)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(C2_OUT, C3_OUT, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(C3_OUT)
        self.relu3 = nn.ReLU(inplace=False)

        self.avgpool = nn.AvgPool2d(kernel_size=16)

        self.fc = nn.Linear(C3_OUT, num_classes)

        self.dequant = tq.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.dequant(x)

        return x

    def fuse_model(self):
        tq.fuse_modules(self, [["conv1", "bn1", "relu1"]], inplace=True)
        tq.fuse_modules(self, [["conv2", "bn2", "relu2"]], inplace=True)
        tq.fuse_modules(self, [["conv3", "bn3", "relu3"]], inplace=True)


# ============================================================
# Load float checkpoint
# ============================================================

def load_float_checkpoint_into_qat(model, checkpoint_path: Path):
    """
    Loads previous SmallFruitCNN checkpoint into the QAT model.

    Expected previous float model:
        features.0  -> conv1
        features.1  -> bn1
        features.4  -> conv2
        features.5  -> bn2
        features.8  -> conv3
        features.9  -> bn3
        classifier  -> fc

    This works if the float model used the same channel counts:
        3 -> 12 -> 24 -> 48 -> 3
    """

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Float checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint["model_state_dict"]

    mapped_state = model.state_dict()

    mapping = {
        "features.0.weight": "conv1.weight",
        "features.1.weight": "bn1.weight",
        "features.1.bias": "bn1.bias",
        "features.1.running_mean": "bn1.running_mean",
        "features.1.running_var": "bn1.running_var",
        "features.1.num_batches_tracked": "bn1.num_batches_tracked",

        "features.4.weight": "conv2.weight",
        "features.5.weight": "bn2.weight",
        "features.5.bias": "bn2.bias",
        "features.5.running_mean": "bn2.running_mean",
        "features.5.running_var": "bn2.running_var",
        "features.5.num_batches_tracked": "bn2.num_batches_tracked",

        "features.8.weight": "conv3.weight",
        "features.9.weight": "bn3.weight",
        "features.9.bias": "bn3.bias",
        "features.9.running_mean": "bn3.running_mean",
        "features.9.running_var": "bn3.running_var",
        "features.9.num_batches_tracked": "bn3.num_batches_tracked",

        "classifier.weight": "fc.weight",
        "classifier.bias": "fc.bias",
    }

    loaded = []
    skipped = []

    for src, dst in mapping.items():
        if src not in state:
            skipped.append((src, dst, "missing in checkpoint"))
            continue

        if dst not in mapped_state:
            skipped.append((src, dst, "missing in QAT model"))
            continue

        if state[src].shape != mapped_state[dst].shape:
            skipped.append((src, dst, f"shape mismatch {state[src].shape} vs {mapped_state[dst].shape}"))
            continue

        mapped_state[dst] = state[src]
        loaded.append((src, dst))

    model.load_state_dict(mapped_state, strict=False)

    print("Loaded float weights:")
    for src, dst in loaded:
        print(f"  {src} -> {dst}")

    if skipped:
        print()
        print("Skipped float weights:")
        for src, dst, reason in skipped:
            print(f"  {src} -> {dst}: {reason}")

    return checkpoint


# ============================================================
# Evaluation / training
# ============================================================

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

        n = labels.size(0)

        total_loss += loss.item() * n
        total_samples += n

    return total_loss / total_samples


# ============================================================
# Main
# ============================================================

def main():
    torch.manual_seed(SEED)
    random.seed(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Programmatic resize.
    # The dataset on disk remains 100x100.
    # The model always receives 64x64.
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
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
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        transforms.ToTensor(),
    ])

    base_dataset = GroupedFruitDataset(DATA_DIR, transform=None)

    train_indices, val_indices = stratified_split(
        base_dataset,
        val_fraction=VAL_FRACTION,
        seed=SEED,
    )

    train_dataset_full = GroupedFruitDataset(DATA_DIR, transform=train_transform)
    val_dataset_full = GroupedFruitDataset(DATA_DIR, transform=val_transform)

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    print("Class names:", CLASS_NAMES)
    print("Total selected images:", len(base_dataset))
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Training class counts:", count_labels(base_dataset, train_indices))
    print("Validation class counts:", count_labels(base_dataset, val_indices))
    print()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    qengine = select_quant_backend()
    torch.backends.quantized.engine = qengine

    print("Training device:", device)
    print("Quantization engine:", qengine)
    print("Model input size:", IMAGE_SIZE, "x", IMAGE_SIZE)
    print()

    model_fp32 = QATFruitCNN64(num_classes=3)

    float_ckpt = load_float_checkpoint_into_qat(model_fp32, FLOAT_CHECKPOINT)

    if "image_size" in float_ckpt:
        ckpt_image_size = int(float_ckpt["image_size"])
        if ckpt_image_size != IMAGE_SIZE:
            print()
            print("WARNING:")
            print(f"  Float checkpoint was trained with image_size={ckpt_image_size}")
            print(f"  Current QAT model uses image_size={IMAGE_SIZE}")
            print("  This can still load because Conv/BN/FC shapes match,")
            print("  but a 64x64 float checkpoint is preferable.")
            print()

    # Fuse before QAT.
    model_fp32.eval()
    model_fp32.fuse_model()

    # Prepare QAT.
    model_fp32.train()
    model_fp32.qconfig = tq.get_default_qat_qconfig(qengine)
    model_qat = tq.prepare_qat(model_fp32, inplace=False)

    model_qat = model_qat.to(device)

    initial_acc, _, _ = evaluate(
        model_qat,
        val_loader,
        device,
        CLASS_NAMES,
    )

    print(f"Initial fake-quant validation accuracy: {initial_acc:.5f}")
    print()

    train_counts = count_labels(base_dataset, train_indices)

    class_weights = torch.tensor(
        [
            1.0 / train_counts["apple"],
            1.0 / train_counts["banana"],
            1.0 / train_counts["orange"],
        ],
        dtype=torch.float32,
    )

    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = class_weights.to(device)

    print("Class weights:", class_weights.detach().cpu().tolist())
    print()

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model_qat.parameters(), lr=LR)

    best_acc = initial_acc
    best_state = copy.deepcopy(model_qat.state_dict())

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model_qat,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_acc, report, cm = evaluate(
            model_qat,
            val_loader,
            device,
            CLASS_NAMES,
        )

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"loss={train_loss:.5f} | "
            f"val_acc={val_acc:.5f}"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model_qat.state_dict())

            torch.save(
                {
                    "model_state_dict": best_state,
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "val_accuracy": best_acc,
                    "architecture": "QATFruitCNN64",
                    "qengine": qengine,
                    "pooling": "AvgPool2d(kernel_size=16)",
                    "channels": [C1_OUT, C2_OUT, C3_OUT],
                    "note": "Prepared QAT fake-quant model state, not converted INT8 model.",
                },
                OUT_DIR / "best_qat_fakequant.pt",
            )

            with open(OUT_DIR / "metrics.txt", "w", encoding="utf-8") as f:
                f.write(f"best_val_accuracy={best_acc:.6f}\n")
                f.write(f"initial_fakequant_accuracy={initial_acc:.6f}\n")
                f.write(f"qengine={qengine}\n")
                f.write(f"image_size={IMAGE_SIZE}\n")
                f.write("pooling=AvgPool2d(kernel_size=16)\n")
                f.write(f"class_weights={class_weights.detach().cpu().tolist()}\n\n")
                f.write(report)
                f.write("\nConfusion matrix:\n")
                f.write(str(cm))
                f.write("\n")

    # Convert to actual CPU quantized model.
    # This is CPU-side PyTorch INT8, not yet FPGA hardware.
    model_qat.load_state_dict(best_state)
    model_qat = model_qat.cpu()
    model_qat.eval()

    model_int8 = tq.convert(model_qat, inplace=False)

    torch.save(
        {
            "model_state_dict": model_int8.state_dict(),
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "val_accuracy": best_acc,
            "architecture": "QATFruitCNN64_INT8",
            "qengine": qengine,
            "pooling": "AvgPool2d(kernel_size=16)",
            "channels": [C1_OUT, C2_OUT, C3_OUT],
            "note": "Converted PyTorch quantized INT8 model state.",
        },
        OUT_DIR / "best_int8_state.pt",
    )

    torch.save(model_int8, OUT_DIR / "best_int8_full_model.pt")

    with open(OUT_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "data_dir": str(DATA_DIR),
                "float_checkpoint": str(FLOAT_CHECKPOINT),
                "out_dir": str(OUT_DIR),
                "image_size": IMAGE_SIZE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "val_fraction": VAL_FRACTION,
                "seed": SEED,
                "qengine": qengine,
                "class_names": CLASS_NAMES,
                "pooling": "AvgPool2d(kernel_size=16)",
                "channels": [C1_OUT, C2_OUT, C3_OUT],
                "resize": "transforms.Resize((64, 64), antialias=True)",
            },
            f,
            indent=2,
        )

    print()
    print(f"Best QAT validation accuracy: {best_acc:.5f}")
    print(f"Saved fake-quant QAT checkpoint: {OUT_DIR / 'best_qat_fakequant.pt'}")
    print(f"Saved converted INT8 state:      {OUT_DIR / 'best_int8_state.pt'}")
    print(f"Saved converted INT8 full model: {OUT_DIR / 'best_int8_full_model.pt'}")
    print(f"Saved metrics:                  {OUT_DIR / 'metrics.txt'}")


if __name__ == "__main__":
    main()
