from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "golden_traces"
OUT_JSON = OUT_DIR / "tileloader_trace.json"
OUT_DDR_HEX = OUT_DIR / "tileloader_ddr.hex"
OUT_A_TILE_HEX = OUT_DIR / "tileloader_A_tile.hex"
OUT_B_TILE_HEX = OUT_DIR / "tileloader_B_tile.hex"
OUT_READS_HEX = OUT_DIR / "tileloader_reads.hex"

ROWS = 32
COLS = 32
TILE = 16
A_BASE = 3 * COLS + 5
B_BASE = 9 * COLS + 11
A_ROW_STRIDE = COLS
B_ROW_STRIDE = COLS


def int8_pattern(index: int) -> int:
    value = ((index * 29 + 17) % 255) - 127
    return max(-128, min(127, value))


def flatten(matrix):
    return [value for row in matrix for value in row]


def write_hex(path: Path, values, width: int):
    mask = (1 << (width * 4)) - 1
    path.write_text("\n".join(f"{value & mask:0{width}x}" for value in values) + "\n", encoding="utf-8")


def build_ddr():
    return [int8_pattern(i) for i in range(ROWS * COLS)]


def tile_from_ddr(ddr, base: int, row_stride: int):
    return [[ddr[base + r * row_stride + c] for c in range(TILE)] for r in range(TILE)]


def build_read_trace(ddr, base: int, row_stride: int, matrix: str, start_cycle: int):
    trace = []
    cycle = start_cycle
    for row in range(TILE):
        for col in range(TILE):
            address = base + row * row_stride + col
            trace.append(
                {
                    "cycle": cycle,
                    "matrix": matrix,
                    "row": row,
                    "col": col,
                    "ddr_addr": address,
                    "cache_addr": row * TILE + col,
                    "data": ddr[address],
                }
            )
            cycle += 1
    return trace


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ddr = build_ddr()
    a_tile = tile_from_ddr(ddr, A_BASE, A_ROW_STRIDE)
    b_tile = tile_from_ddr(ddr, B_BASE, B_ROW_STRIDE)
    a_reads = build_read_trace(ddr, A_BASE, A_ROW_STRIDE, "A", 0)
    b_reads = build_read_trace(ddr, B_BASE, B_ROW_STRIDE, "B", len(a_reads))
    reads = a_reads + b_reads

    OUT_JSON.write_text(
        json.dumps(
            {
                "rows": ROWS,
                "cols": COLS,
                "tile": TILE,
                "a_base": A_BASE,
                "b_base": B_BASE,
                "a_row_stride": A_ROW_STRIDE,
                "b_row_stride": B_ROW_STRIDE,
                "reads": reads,
                "A_tile": a_tile,
                "B_tile": b_tile,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_hex(OUT_DDR_HEX, ddr, 2)
    write_hex(OUT_A_TILE_HEX, flatten(a_tile), 2)
    write_hex(OUT_B_TILE_HEX, flatten(b_tile), 2)
    write_hex(
        OUT_READS_HEX,
        [
            ((0 if row["matrix"] == "A" else 1) << 40)
            | ((row["ddr_addr"] & 0xFFFF) << 24)
            | ((row["cache_addr"] & 0xFFFF) << 8)
            | (row["data"] & 0xFF)
            for row in reads
        ],
        11,
    )

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_DDR_HEX}")
    print(f"Wrote {OUT_A_TILE_HEX}")
    print(f"Wrote {OUT_B_TILE_HEX}")
    print(f"Wrote {OUT_READS_HEX}")


if __name__ == "__main__":
    main()
