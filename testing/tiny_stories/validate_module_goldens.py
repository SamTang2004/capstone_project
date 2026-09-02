"""Validate every archive and tensor in a W4A8 module-golden manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "FPGA"))

from budgetcuda.stories_module_goldens import validate_w4a8_module_goldens  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_w4a8_module_goldens(args.manifest), indent=2))


if __name__ == "__main__":
    main()
