from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "golden_traces"
OUT_JSON = OUT_DIR / "systolic_4x4_trace.json"
OUT_A_HEX = OUT_DIR / "systolic_4x4_A.hex"
OUT_B_HEX = OUT_DIR / "systolic_4x4_B.hex"
OUT_C_HEX = OUT_DIR / "systolic_4x4_C.hex"


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


def write_int8_hex(path: Path, matrix):
    path.write_text("\n".join(f"{value & 0xff:02x}" for value in flatten(matrix)) + "\n", encoding="utf-8")


def write_int32_hex(path: Path, matrix):
    path.write_text("\n".join(f"{value & 0xffffffff:08x}" for value in flatten(matrix)) + "\n", encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    A = [
        [1, 2, -3, 4],
        [-5, 6, 7, -8],
        [9, -10, 11, 12],
        [-13, 14, -15, 16],
    ]
    B = [
        [2, -1, 3, 4],
        [5, 6, -7, 8],
        [-9, 10, 11, -12],
        [13, -14, 15, 16],
    ]
    C = matmul(A, B)
    OUT_JSON.write_text(json.dumps({"A": A, "B": B, "C": C}, indent=2), encoding="utf-8")
    write_int8_hex(OUT_A_HEX, A)
    write_int8_hex(OUT_B_HEX, B)
    write_int32_hex(OUT_C_HEX, C)
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_A_HEX}")
    print(f"Wrote {OUT_B_HEX}")
    print(f"Wrote {OUT_C_HEX}")


if __name__ == "__main__":
    main()
