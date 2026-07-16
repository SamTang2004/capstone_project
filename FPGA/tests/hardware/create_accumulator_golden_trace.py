from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "golden_traces"
OUT_JSON = OUT_DIR / "accumulator_trace.json"
OUT_OPS_HEX = OUT_DIR / "accumulator_ops.hex"
OUT_FINAL_HEX = OUT_DIR / "accumulator_final_tile.hex"
OUT_READS_HEX = OUT_DIR / "accumulator_reads.hex"

TILE = 16


def partial_value(pass_id: int, row: int, col: int) -> int:
    value = (pass_id + 1) * 97 + (row - 7) * 23 - (col - 5) * 19
    return -value if (pass_id + row + col) % 4 == 0 else value


def flatten(matrix: list[list[int]]) -> list[int]:
    return [value for row in matrix for value in row]


def write_hex(path: Path, values: list[int], width: int) -> None:
    mask = (1 << (width * 4)) - 1
    path.write_text("\n".join(f"{value & mask:0{width}x}" for value in values) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    accumulator = [[0 for _ in range(TILE)] for _ in range(TILE)]
    ops: list[dict] = []
    reads: list[dict] = []
    cycle = 0

    for pass_id in range(3):
        for row in range(TILE):
            for col in range(TILE):
                addr = row * TILE + col
                partial = partial_value(pass_id, row, col)
                before = accumulator[row][col]
                if pass_id == 0:
                    before = 0
                    accumulator[row][col] = partial
                else:
                    accumulator[row][col] += partial

                ops.append(
                    {
                        "cycle": cycle,
                        "pass": pass_id,
                        "row": row,
                        "col": col,
                        "addr": addr,
                        "clear": pass_id == 0,
                        "partial": partial,
                        "before": before,
                        "after": accumulator[row][col],
                    }
                )
                cycle += 1

    for row in range(TILE):
        for col in range(TILE):
            addr = row * TILE + col
            reads.append(
                {
                    "cycle": cycle,
                    "row": row,
                    "col": col,
                    "addr": addr,
                    "data": accumulator[row][col],
                }
            )
            cycle += 1

    OUT_JSON.write_text(
        json.dumps(
            {
                "tile": TILE,
                "passes": 3,
                "op_count": len(ops),
                "read_count": len(reads),
                "ops": ops,
                "reads": reads,
                "final_tile": accumulator,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_hex(
        OUT_OPS_HEX,
        [
            ((op["clear"] & 0x1) << 56)
            | ((op["addr"] & 0xFF) << 48)
            | ((op["partial"] & 0xFFFFFFFF) << 16)
            | (op["after"] & 0xFFFF)
            for op in ops
        ],
        16,
    )
    write_hex(OUT_FINAL_HEX, flatten(accumulator), 8)
    write_hex(
        OUT_READS_HEX,
        [((read["addr"] & 0xFF) << 32) | (read["data"] & 0xFFFFFFFF) for read in reads],
        10,
    )

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_OPS_HEX}")
    print(f"Wrote {OUT_FINAL_HEX}")
    print(f"Wrote {OUT_READS_HEX}")


if __name__ == "__main__":
    main()
