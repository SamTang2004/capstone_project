from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "golden_traces"
OUT_JSON = OUT_DIR / "memory_io_trace.json"
OUT_INITIAL_DDR_HEX = OUT_DIR / "memory_io_initial_ddr.hex"
OUT_READS_HEX = OUT_DIR / "memory_io_reads.hex"
OUT_WRITES_HEX = OUT_DIR / "memory_io_writes.hex"
OUT_FINAL_DDR_HEX = OUT_DIR / "memory_io_final_ddr.hex"

MEM_BYTES = 256


def initial_byte(index: int) -> int:
    return (0x31 + index * 17 + (index >> 2) * 9) & 0xFF


def write_hex(path: Path, values: list[int], width: int) -> None:
    mask = (1 << (width * 4)) - 1
    path.write_text("\n".join(f"{value & mask:0{width}x}" for value in values) + "\n", encoding="utf-8")


def read_byte(ddr: list[int], addr: int) -> int:
    return ddr[addr]


def write_word_le(ddr: list[int], addr: int, data: int) -> None:
    if addr & 0x3:
        raise ValueError(f"INT32 write address must be aligned: {addr}")
    for lane in range(4):
        ddr[addr + lane] = (data >> (8 * lane)) & 0xFF


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    initial_ddr = [initial_byte(i) for i in range(MEM_BYTES)]
    final_ddr = initial_ddr[:]

    read_plan = [
        {"cycle": 2, "addr": 0x03, "stall_response_cycles": 0},
        {"cycle": 5, "addr": 0x10, "stall_response_cycles": 2},
        {"cycle": 8, "addr": 0x21, "stall_response_cycles": 0},
        {"cycle": 9, "addr": 0x22, "stall_response_cycles": 1},
        {"cycle": 14, "addr": 0x40, "stall_response_cycles": 0},
        {"cycle": 18, "addr": 0x43, "stall_response_cycles": 0},
    ]
    write_plan = [
        {"cycle": 5, "addr": 0x20, "data": 0x11223344},
        {"cycle": 9, "addr": 0x40, "data": 0xA5B6C7D8},
        {"cycle": 13, "addr": 0x7C, "data": 0x01020304},
        {"cycle": 18, "addr": 0x44, "data": 0xFEEDC0DE},
    ]

    reads = []
    for read in read_plan:
        visible_ddr = initial_ddr[:]
        for write in write_plan:
            if write["cycle"] <= read["cycle"]:
                write_word_le(visible_ddr, write["addr"], write["data"])
        aligned_addr = read["addr"] & ~0x3
        reads.append(
            {
                **read,
                "aligned_addr": aligned_addr,
                "lane": read["addr"] & 0x3,
                "data": read_byte(visible_ddr, read["addr"]),
            }
        )

    for write in write_plan:
        write_word_le(final_ddr, write["addr"], write["data"])

    writes = [
        {
            **write,
            "wstrb": 0xF,
            "bytes": [(write["data"] >> (8 * lane)) & 0xFF for lane in range(4)],
        }
        for write in write_plan
    ]

    OUT_JSON.write_text(
        json.dumps(
            {
                "mem_bytes": MEM_BYTES,
                "endianness": "little",
                "write_priority": True,
                "initial_ddr": initial_ddr,
                "reads": reads,
                "writes": writes,
                "final_ddr": final_ddr,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_hex(OUT_INITIAL_DDR_HEX, initial_ddr, 2)
    write_hex(
        OUT_READS_HEX,
        [
            ((read["cycle"] & 0xFFFF) << 48)
            | ((read["addr"] & 0xFFFFFFFF) << 16)
            | ((read["stall_response_cycles"] & 0xFF) << 8)
            | (read["data"] & 0xFF)
            for read in reads
        ],
        16,
    )
    write_hex(
        OUT_WRITES_HEX,
        [
            ((write["cycle"] & 0xFFFF) << 80)
            | ((write["addr"] & 0xFFFFFFFF) << 48)
            | ((write["data"] & 0xFFFFFFFF) << 16)
            | (write["wstrb"] & 0xF)
            for write in writes
        ],
        24,
    )
    write_hex(OUT_FINAL_DDR_HEX, final_ddr, 2)

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_INITIAL_DDR_HEX}")
    print(f"Wrote {OUT_READS_HEX}")
    print(f"Wrote {OUT_WRITES_HEX}")
    print(f"Wrote {OUT_FINAL_DDR_HEX}")


if __name__ == "__main__":
    main()
