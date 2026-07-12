from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "golden_traces"
OUT_JSON = OUT_DIR / "pe_int8_mac_trace.json"
OUT_HEX = OUT_DIR / "pe_int8_mac_trace.hex"


def to_int8(value: int) -> int:
    value &= 0xFF
    return value - 256 if value >= 128 else value


def to_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


def u32(value: int) -> int:
    return value & 0xFFFFFFFF


def build_trace():
    vectors = [
        {"clear": 1, "enable": 0, "a": 0, "b": 0},
        {"clear": 0, "enable": 1, "a": 3, "b": 4},
        {"clear": 0, "enable": 1, "a": -7, "b": 5},
        {"clear": 0, "enable": 1, "a": -8, "b": -9},
        {"clear": 0, "enable": 0, "a": 99, "b": 99},
        {"clear": 0, "enable": 1, "a": 127, "b": -128},
        {"clear": 1, "enable": 1, "a": 2, "b": 2},
        {"clear": 0, "enable": 1, "a": 2, "b": 2},
        {"clear": 0, "enable": 0, "a": 0, "b": 0},
    ]
    acc = 0
    product = 0
    product_valid = False
    trace = []
    for cycle, vector in enumerate(vectors):
        if vector["clear"]:
            acc = 0
            product = 0
            product_valid = False
        else:
            if product_valid:
                acc = to_int32(acc + product)
            if vector["enable"]:
                product = to_int8(vector["a"]) * to_int8(vector["b"])
                product_valid = True
            else:
                product_valid = False
        trace.append({**vector, "cycle": cycle, "expected_acc": acc})
    return trace


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trace = build_trace()
    OUT_JSON.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    lines = [
        f"{row['clear'] & 1:01x}{row['enable'] & 1:01x}{to_int8(row['a']) & 0xff:02x}{to_int8(row['b']) & 0xff:02x}{u32(row['expected_acc']):08x}"
        for row in trace
    ]
    OUT_HEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_HEX}")


if __name__ == "__main__":
    main()
