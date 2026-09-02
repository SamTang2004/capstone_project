"""Validate a TinyStories JSON/NPZ golden trace pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "FPGA"))

from budgetcuda.stories import validate_stories_golden  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path, help="golden JSON index")
    args = parser.parse_args()
    print(json.dumps(validate_stories_golden(args.index), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

