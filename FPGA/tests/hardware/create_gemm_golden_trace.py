from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "golden_traces"
OUT_JSON = OUT_DIR / "gemm_16x16_trace.json"
OUT_A_HEX = OUT_DIR / "gemm_16x16_A.hex"
OUT_B_HEX = OUT_DIR / "gemm_16x16_B.hex"
OUT_C_HEX = OUT_DIR / "gemm_16x16_C.hex"


def int8_pattern(index: int) -> int:
    value = ((index * 37 + 19) % 255) - 127
    return max(-128, min(127, value))


def matrix(rows: int, cols: int, offset: int):
    return [[int8_pattern(offset + r * cols + c) for c in range(cols)] for r in range(rows)]


def matmul(A, B):
    rows = len(A)
    cols = len(B[0])
    inner = len(B)
    return [
        [sum(int(A[r][k]) * int(B[k][c]) for k in range(inner)) for c in range(cols)]
        for r in range(rows)
    ]


def flatten(matrix):
    return [value for row in matrix for value in row]


def write_hex(path: Path, values, width: int):
    mask = (1 << (width * 4)) - 1
    path.write_text("\n".join(f"{value & mask:0{width}x}" for value in values) + "\n", encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    A = matrix(16, 16, 0)
    B = matrix(16, 16, 4096)
    C = matmul(A, B)
    OUT_JSON.write_text(json.dumps({"M": 16, "N": 16, "K": 16, "A": A, "B": B, "C": C}, indent=2), encoding="utf-8")
    write_hex(OUT_A_HEX, flatten(A), 2)
    write_hex(OUT_B_HEX, flatten(B), 2)
    write_hex(OUT_C_HEX, flatten(C), 8)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_A_HEX}")
    print(f"Wrote {OUT_B_HEX}")
    print(f"Wrote {OUT_C_HEX}")


if __name__ == "__main__":
    main()
