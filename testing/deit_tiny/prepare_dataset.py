"""Validate the selected Mendeley fruit classes and create grouped splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps


CLASS_FOLDERS = {
    "apple_fresh": "FreshApple",
    "apple_rotten": "RottenApple",
    "banana_fresh": "FreshBanana",
    "banana_rotten": "RottenBanana",
    "orange_fresh": "FreshOrange",
    "orange_rotten": "RottenOrange",
}
CLASS_NAMES = tuple(CLASS_FOLDERS)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
INDEX_PATTERN = re.compile(r"\((\d+)\)")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=here / "data" / "original" / "Original Image",
    )
    parser.add_argument("--output", type=Path, default=here / "splits" / "manifest.csv")
    parser.add_argument("--cache-root", type=Path, default=here / "data" / "cache_256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--group-size",
        type=int,
        default=5,
        help="Keep adjacent source-image indices together to reduce capture-sequence leakage.",
    )
    return parser.parse_args()


def numeric_index(path: Path) -> int:
    match = INDEX_PATTERN.search(path.stem)
    if not match:
        raise ValueError(f"Cannot extract source index from {path.name}")
    return int(match.group(1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assign_group_splits(group_ids: list[int], seed: int) -> dict[int, str]:
    shuffled = list(group_ids)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    validation_count = max(1, round(count * 0.15))
    test_count = max(1, round(count * 0.15))
    train_count = count - validation_count - test_count
    assignments: dict[int, str] = {}
    for group_id in shuffled[:train_count]:
        assignments[group_id] = "train"
    for group_id in shuffled[train_count : train_count + validation_count]:
        assignments[group_id] = "validation"
    for group_id in shuffled[train_count + validation_count :]:
        assignments[group_id] = "test"
    return assignments


def cache_image(source: Path, destination: Path, short_edge: int = 256) -> None:
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        scale = short_edge / min(width, height)
        resized = image.resize(
            (round(width * scale), round(height * scale)), Image.Resampling.BICUBIC
        )
        resized.save(destination, format="JPEG", quality=95, subsampling=0)


def main() -> None:
    args = parse_args()
    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {args.data_root}")

    records: list[dict[str, str | int]] = []
    seen_hashes: dict[str, Path] = {}
    duplicates: list[tuple[Path, Path]] = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        folder = args.data_root / CLASS_FOLDERS[class_name]
        paths = sorted(
            (path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS),
            key=numeric_index,
        )
        if not paths:
            raise RuntimeError(f"No images found in {folder}")

        group_ids = sorted({(numeric_index(path) - 1) // args.group_size for path in paths})
        assignments = assign_group_splits(group_ids, args.seed + class_index)
        for path in paths:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
            digest = sha256(path)
            if digest in seen_hashes:
                duplicates.append((seen_hashes[digest], path))
                continue
            seen_hashes[digest] = path
            source_index = numeric_index(path)
            capture_group = (source_index - 1) // args.group_size
            relative_path = path.relative_to(args.data_root)
            cache_image(path, args.cache_root / relative_path)
            records.append(
                {
                    "relative_path": relative_path.as_posix(),
                    "class_name": class_name,
                    "class_index": class_index,
                    "split": assignments[capture_group],
                    "source_index": source_index,
                    "capture_group": capture_group,
                    "width": width,
                    "height": height,
                    "sha256": digest,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0])
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    split_counts = Counter(record["split"] for record in records)
    class_split_counts = Counter(
        f"{record['class_name']}:{record['split']}" for record in records
    )
    summary = {
        "dataset": "Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality",
        "doi": "10.17632/bdd69gyhv8.1",
        "license": "CC BY 4.0",
        "source_archive_sha256": "f89d67d4c4b24810bcd8406db877db6b1ac9834cea0b0cb56d65acebe59e66ff",
        "seed": args.seed,
        "group_size": args.group_size,
        "classes": list(CLASS_NAMES),
        "total_images": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "class_split_counts": dict(sorted(class_split_counts.items())),
        "exact_duplicates_excluded": len(duplicates),
        "excluded_duplicate_paths": [
            duplicate.relative_to(args.data_root).as_posix() for _, duplicate in duplicates
        ],
    }
    summary_path = args.output.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
