"""Create the deterministic balanced Fruits-360 subset used by the FPGA demo."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import shutil
import sys


FPGA_ROOT = Path(__file__).resolve().parents[1]
TRAIN_MODEL = FPGA_ROOT / "train_model"
sys.path.insert(0, str(TRAIN_MODEL))

from paths import DATASET_ROOT
from qat_model import CLASS_NAMES, GroupedFruitDataset


DEFAULT_OUTPUT = FPGA_ROOT / "demo" / "dataset" / "fruits-360-100x100"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.per_class <= 0:
        raise ValueError("per-class must be positive")
    if args.output_root.exists():
        if not args.force:
            raise FileExistsError(f"output exists; pass --force to replace it: {args.output_root}")
        shutil.rmtree(args.output_root)

    source_test = args.source_root / "Test"
    dataset = GroupedFruitDataset(source_test)
    grouped: dict[int, list[tuple[Path, int, str]]] = defaultdict(list)
    for sample in dataset.samples:
        grouped[sample[1]].append(sample)

    rng = random.Random(args.seed)
    selected = []
    for target, class_name in enumerate(CLASS_NAMES):
        candidates = grouped[target]
        if len(candidates) < args.per_class:
            raise RuntimeError(f"class {class_name} has only {len(candidates)} images")
        selected.extend(rng.sample(candidates, args.per_class))
    rng.shuffle(selected)

    manifest = {
        "source": "Fruits-360 100x100",
        "seed": args.seed,
        "per_class": args.per_class,
        "classes": CLASS_NAMES,
        "files": [],
    }
    output_test = args.output_root / "Test"
    for path, target, folder in selected:
        relative = path.relative_to(source_test)
        destination = output_test / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        manifest["files"].append(
            {
                "path": destination.relative_to(args.output_root).as_posix(),
                "class": CLASS_NAMES[target],
                "source_folder": folder,
                "sha256": sha256(destination),
            }
        )

    for name in ("LICENSE", "README.md"):
        source = args.source_root / name
        if source.exists():
            shutil.copy2(source, args.output_root / name)

    manifest_path = args.output_root / "demo_subset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    total_bytes = sum(path.stat().st_size for path in output_test.rglob("*") if path.is_file())
    print(
        f"created {len(selected)} images at {args.output_root} "
        f"({total_bytes} bytes, seed={args.seed}, per_class={args.per_class})"
    )


if __name__ == "__main__":
    main()
