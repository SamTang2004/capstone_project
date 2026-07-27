#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pynq import Overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bitfile", type=Path)
    args = parser.parse_args()

    bitfile = args.bitfile.resolve()
    hwh = bitfile.with_suffix(".hwh")
    if not bitfile.is_file() or not hwh.is_file():
        raise FileNotFoundError(f"overlay pair is incomplete: {bitfile}, {hwh}")

    overlay = Overlay(str(bitfile), download=True)
    if not overlay.is_loaded():
        raise RuntimeError("BudgetCUDA overlay did not report loaded")
    if "PSDDR" not in overlay.mem_dict:
        raise RuntimeError("BudgetCUDA overlay does not expose PSDDR")

    memory = overlay.mem_dict["PSDDR"]
    print(
        f"BudgetCUDA overlay loaded: PSDDR base=0x{memory['base_address']:x} "
        f"size={memory['size']}"
    )


if __name__ == "__main__":
    main()
