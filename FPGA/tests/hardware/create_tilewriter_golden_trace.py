from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "golden_traces"
OUT_JSON = OUT_DIR / "tilewriter_trace.json"
OUT_INITIAL_DDR_HEX = OUT_DIR / "tilewriter_initial_ddr.hex"
OUT_C_TILE_HEX = OUT_DIR / "tilewriter_C_tile.hex"
OUT_WRITES_HEX = OUT_DIR / "tilewriter_writes.hex"
OUT_FINAL_DDR_HEX = OUT_DIR / "tilewriter_final_ddr.hex"

ROWS = 32
COLS = 32
TILE = 16
C_BASE = 7 * COLS + 3
C_ROW_STRIDE = COLS


def initial_word(index: int) -> int:
    return 0x40000000 + index


def c_value(row: int, col: int) -> int:
    value = (row - 8) * 257 + (col - 5) * 31
    return value if (row + col) % 3 else -value


def flatten(matrix):
    return [value for row in matrix for value in row]


def write_hex(path: Path, values, width: int):
    mask = (1 << (width * 4)) - 1
    path.write_text("\n".join(f"{value & mask:0{width}x}" for value in values) + "\n", encoding="utf-8")


def build_write_trace(c_tile):
    trace = []
    cycle = 0
    for row in range(TILE):
        for col in range(TILE):
            trace.append(
                {
                    "cycle": cycle,
                    "row": row,
                    "col": col,
                    "ddr_addr": C_BASE + row * C_ROW_STRIDE + col,
                    "tile_addr": row * TILE + col,
                    "data": c_tile[row][col],
                }
            )
            cycle += 1
    return trace


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    initial_ddr = [initial_word(i) for i in range(ROWS * COLS)]
    final_ddr = initial_ddr[:]
    c_tile = [[c_value(row, col) for col in range(TILE)] for row in range(TILE)]
    writes = build_write_trace(c_tile)

    for row in writes:
        final_ddr[row["ddr_addr"]] = row["data"]

    OUT_JSON.write_text(
        json.dumps(
            {
                "rows": ROWS,
                "cols": COLS,
                "tile": TILE,
                "c_base": C_BASE,
                "c_row_stride": C_ROW_STRIDE,
                "writes": writes,
                "C_tile": c_tile,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_hex(OUT_INITIAL_DDR_HEX, initial_ddr, 8)
    write_hex(OUT_C_TILE_HEX, flatten(c_tile), 8)
    write_hex(
        OUT_WRITES_HEX,
        [
            ((row["ddr_addr"] & 0xFFFF) << 48)
            | ((row["tile_addr"] & 0xFFFF) << 32)
            | (row["data"] & 0xFFFFFFFF)
            for row in writes
        ],
        16,
    )
    write_hex(OUT_FINAL_DDR_HEX, final_ddr, 8)

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_INITIAL_DDR_HEX}")
    print(f"Wrote {OUT_C_TILE_HEX}")
    print(f"Wrote {OUT_WRITES_HEX}")
    print(f"Wrote {OUT_FINAL_DDR_HEX}")


if __name__ == "__main__":
    main()
