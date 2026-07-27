from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


SIZE = 8
OUT_DIR = Path(__file__).resolve().parent / "golden_traces"


def matmul(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [
        [sum(a[row][k] * b[k][col] for k in range(SIZE)) for col in range(SIZE)]
        for row in range(SIZE)
    ]


def flatten(matrix: Iterable[Iterable[int]]) -> list[int]:
    return [value for row in matrix for value in row]


def write_int4_hex(name: str, matrix: list[list[int]]) -> None:
    values = flatten(matrix)
    assert all(-8 <= value <= 7 for value in values)
    (OUT_DIR / name).write_text(
        "\n".join(f"{value & 0xF:x}" for value in values) + "\n",
        encoding="ascii",
    )


def write_int32_hex(name: str, matrix: list[list[int]]) -> None:
    (OUT_DIR / name).write_text(
        "\n".join(f"{value & 0xFFFFFFFF:08x}" for value in flatten(matrix)) + "\n",
        encoding="ascii",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    a0 = [[((row * 5 + k * 3 + 1) % 16) - 8 for k in range(SIZE)] for row in range(SIZE)]
    b0 = [[((k * 7 - col * 5 + 2) % 16) - 8 for col in range(SIZE)] for k in range(SIZE)]
    a1 = [[((row * 3 - k * 7 + 6) % 16) - 8 for k in range(SIZE)] for row in range(SIZE)]
    b1 = [[((k * 5 + col * 7 + 4) % 16) - 8 for col in range(SIZE)] for k in range(SIZE)]

    outputs = {
        "int4_8x8_C0.hex": matmul(a0, b0),
        "int4_8x8_C1_independent.hex": matmul(a1, b1),
        "int4_8x8_C1_shared_a.hex": matmul(a0, b1),
        "int4_8x8_C1_shared_b.hex": matmul(a1, b0),
    }

    write_int4_hex("int4_8x8_A0.hex", a0)
    write_int4_hex("int4_8x8_B0.hex", b0)
    write_int4_hex("int4_8x8_A1.hex", a1)
    write_int4_hex("int4_8x8_B1.hex", b1)
    for name, matrix in outputs.items():
        write_int32_hex(name, matrix)

    metadata = {
        "format": "signed two's-complement INT4",
        "shape": [SIZE, SIZE],
        "accumulator": "signed INT32",
        "share_modes": {
            "independent": "C0=A0*B0, C1=A1*B1",
            "shared_a": "C0=A0*B0, C1=A0*B1",
            "shared_b": "C0=A0*B0, C1=A1*B0",
        },
    }
    (OUT_DIR / "int4_8x8_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="ascii",
    )
    print(f"Wrote signed INT4 golden traces to {OUT_DIR}")


if __name__ == "__main__":
    main()
