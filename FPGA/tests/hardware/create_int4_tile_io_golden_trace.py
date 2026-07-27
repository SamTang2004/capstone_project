from __future__ import annotations

import json
from pathlib import Path


TILE = 8
DDR_SIZE = 256
A_BASE = 17
B_BASE = 83
INPUT_ROW_STRIDE = 5
C_BASE = 151
C_ROW_STRIDE = 5
C_COLS = 7
OUT_DIR = Path(__file__).resolve().parent / "golden_traces"


def a_value(row: int, col: int) -> int:
    return ((row * 5 + col * 3 + 1) % 16) - 8


def b_value(row: int, col: int) -> int:
    return ((row * 7 - col * 5 + 2) % 16) - 8


def c_value(row: int, col: int) -> int:
    return ((row * 3 + col * 7 + 5) % 16) - 8


def pack_row(values: list[int]) -> list[int]:
    packed = []
    for col in range(0, len(values), 2):
        low = values[col] & 0xF
        high = values[col + 1] & 0xF if col + 1 < len(values) else 0
        packed.append(low | (high << 4))
    return packed


def write_hex(path: Path, values: list[int], digits: int) -> None:
    mask = (1 << (digits * 4)) - 1
    path.write_text(
        "\n".join(f"{value & mask:0{digits}x}" for value in values) + "\n",
        encoding="ascii",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ddr = [0xCC] * DDR_SIZE
    a_tile = [[a_value(row, col) for col in range(TILE)] for row in range(TILE)]
    b_tile = [[b_value(row, col) for col in range(TILE)] for row in range(TILE)]
    c_tile = [
        [c_value(row, col) if col < C_COLS else 0 for col in range(TILE)]
        for row in range(TILE)
    ]

    read_trace = []
    for matrix, base, tile in ((0, A_BASE, a_tile), (1, B_BASE, b_tile)):
        for row in range(TILE):
            packed = pack_row(tile[row])
            for byte_col, byte in enumerate(packed):
                address = base + row * INPUT_ROW_STRIDE + byte_col
                ddr[address] = byte
                read_trace.append((matrix << 24) | (address << 8) | byte)

    writer_bytes = []
    writer_trace = []
    for row in range(TILE):
        for byte_col, byte in enumerate(pack_row(c_tile[row][:C_COLS])):
            address = C_BASE + row * C_ROW_STRIDE + byte_col
            writer_bytes.append(byte)
            writer_trace.append((address << 8) | byte)

    write_hex(OUT_DIR / "int4_tileio_ddr.hex", ddr, 2)
    write_hex(OUT_DIR / "int4_tileio_A.hex", [v for row in a_tile for v in row], 1)
    write_hex(OUT_DIR / "int4_tileio_B.hex", [v for row in b_tile for v in row], 1)
    write_hex(OUT_DIR / "int4_tileio_C.hex", [v for row in c_tile for v in row], 1)
    write_hex(OUT_DIR / "int4_tileio_reads.hex", read_trace, 7)
    write_hex(OUT_DIR / "int4_tileio_writer_bytes.hex", writer_bytes, 2)
    write_hex(OUT_DIR / "int4_tileio_writes.hex", writer_trace, 6)

    metadata = {
        "format": "signed two's-complement INT4",
        "nibble_order": "even column low, odd column high",
        "row_alignment": "each row starts at a byte boundary",
        "unused_tail_nibble": 0,
        "tile": TILE,
        "a_base": A_BASE,
        "b_base": B_BASE,
        "input_row_stride_bytes": INPUT_ROW_STRIDE,
        "c_base": C_BASE,
        "c_cols": C_COLS,
        "c_row_stride_bytes": C_ROW_STRIDE,
        "input_read_bytes": len(read_trace),
        "output_write_bytes": len(writer_trace),
    }
    (OUT_DIR / "int4_tileio_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="ascii",
    )
    print(f"Wrote packed INT4 tile I/O traces to {OUT_DIR}")


if __name__ == "__main__":
    main()
