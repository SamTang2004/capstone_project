from __future__ import annotations

import json
from pathlib import Path


TILE_ROWS = 8
TILE_BANKS = 4
DESCRIPTOR_COUNT = 6
OUT_DIR = Path(__file__).resolve().parent / "golden_traces"


def memory_byte(address: int) -> int:
    return ((address * 29) ^ (address >> 3) ^ 0xA7) & 0xFF


def memory_word(address: int) -> int:
    return sum(memory_byte(address + lane) << (8 * lane) for lane in range(4))


def mask_int4_lanes(word: int, mask: int) -> int:
    result = 0
    for lane in range(8):
        if mask & (1 << lane):
            result |= ((word >> (4 * lane)) & 0xF) << (4 * lane)
    return result


def descriptor(index: int) -> dict[str, int]:
    base = 0x1000 + index * 0x200
    return {
        "id": 0x40 + index,
        "a_base": base + index % 4,
        "b0_base": base + 0x80 + (index + 1) % 4,
        "b1_base": base + 0x100 + (index + 2) % 4,
        "a_stride": 4 + index % 4,
        "b0_stride": 4 + (index + 1) % 4,
        "b1_stride": 4 + (index + 2) % 4,
        "a_row_mask": (1 << (8 - index % 3)) - 1,
        "b_row_mask": (1 << (8 - (index + 1) % 3)) - 1,
        "a_lane_mask": (1 << (8 - index % 4)) - 1,
        "b0_lane_mask": (1 << (8 - (index + 1) % 4)) - 1,
        "b1_lane_mask": (1 << (8 - (index + 2) % 4)) - 1,
    }


def pack_descriptor(desc: dict[str, int]) -> int:
    value = desc["id"] << 240
    value |= desc["a_base"] << 208
    value |= desc["b0_base"] << 176
    value |= desc["b1_base"] << 144
    value |= desc["a_stride"] << 128
    value |= desc["b0_stride"] << 112
    value |= desc["b1_stride"] << 96
    value |= desc["a_row_mask"] << 88
    value |= desc["b_row_mask"] << 80
    value |= desc["a_lane_mask"] << 72
    value |= desc["b0_lane_mask"] << 64
    value |= desc["b1_lane_mask"] << 56
    return value


def pack_output(
    desc_id: int,
    row: int,
    a_mask: int,
    b0_mask: int,
    b1_mask: int,
    a_data: int,
    b0_data: int,
    b1_data: int,
) -> int:
    value = desc_id << 160
    value |= row << 157
    value |= a_mask << 149
    value |= b0_mask << 141
    value |= b1_mask << 133
    value |= b1_data << 64
    value |= b0_data << 32
    value |= a_data
    return value


def write_hex(path: Path, values: list[int], digits: int) -> None:
    mask = (1 << (digits * 4)) - 1
    path.write_text(
        "\n".join(f"{value & mask:0{digits}x}" for value in values) + "\n",
        encoding="ascii",
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    descriptors = [descriptor(index) for index in range(DESCRIPTOR_COUNT)]
    requests: list[int] = []
    outputs: list[int] = []

    for desc in descriptors:
        for base_name, stride_name in (
            ("a_base", "a_stride"),
            ("b0_base", "b0_stride"),
            ("b1_base", "b1_stride"),
        ):
            for row in range(TILE_ROWS):
                logical_address = (
                    desc[base_name] + row * desc[stride_name]
                )
                aligned_address = logical_address & ~0x3
                requests.append(aligned_address)
                if logical_address & 0x3:
                    requests.append(aligned_address + 4)

        for row in range(TILE_ROWS):
            a_mask = desc["a_lane_mask"] if desc["a_row_mask"] & (1 << row) else 0
            b0_mask = (
                desc["b0_lane_mask"] if desc["b_row_mask"] & (1 << row) else 0
            )
            b1_mask = (
                desc["b1_lane_mask"] if desc["b_row_mask"] & (1 << row) else 0
            )
            a_data = mask_int4_lanes(
                memory_word(desc["a_base"] + row * desc["a_stride"]), a_mask
            )
            b0_data = mask_int4_lanes(
                memory_word(desc["b0_base"] + row * desc["b0_stride"]), b0_mask
            )
            b1_data = mask_int4_lanes(
                memory_word(desc["b1_base"] + row * desc["b1_stride"]), b1_mask
            )
            outputs.append(
                pack_output(
                    desc["id"],
                    row,
                    a_mask,
                    b0_mask,
                    b1_mask,
                    a_data,
                    b0_data,
                    b1_data,
                )
            )

    write_hex(
        OUT_DIR / "lookahead_tileloader_descriptors.hex",
        [pack_descriptor(desc) for desc in descriptors],
        64,
    )
    write_hex(OUT_DIR / "lookahead_tileloader_requests.hex", requests, 8)
    write_hex(OUT_DIR / "lookahead_tileloader_outputs.hex", outputs, 44)

    metadata = {
        "format": "BudgetCUDA deterministic look-ahead INT4 tile stream v2",
        "tile_banks": TILE_BANKS,
        "descriptor_count": DESCRIPTOR_COUNT,
        "words_per_tile": 24,
        "rows_per_tile": TILE_ROWS,
        "request_count": len(requests),
        "output_count": len(outputs),
        "descriptor_order": [
            "A rows 0..7",
            "B0 rows 0..7",
            "B1 rows 0..7",
        ],
        "output_hex_layout": {
            "175:160": "descriptor_id",
            "159:157": "row",
            "156:149": "A lane mask",
            "148:141": "B0 lane mask",
            "140:133": "B1 lane mask",
            "95:64": "B1 data",
            "63:32": "B0 data",
            "31:0": "A data",
        },
        "requirements": [
            "Descriptors and outputs remain ordered.",
            "At most four tile descriptors are allocated at once.",
            "DDR and output backpressure must not alter payloads.",
            "Unaligned logical vectors are gathered from two aligned DDR words.",
            "Inactive rows and lanes are emitted as masked zeroes.",
            "The descriptor source must stall while all tile credits are allocated.",
        ],
        "descriptors": descriptors,
    }
    (OUT_DIR / "lookahead_tileloader_trace.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="ascii",
    )
    print(
        "Wrote look-ahead TileLoader golden trace: "
        f"{len(descriptors)} descriptors, {len(requests)} reads, "
        f"{len(outputs)} output rows"
    )


if __name__ == "__main__":
    main()
