from __future__ import annotations

import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "golden_traces"
OUT_JSON = OUT_DIR / "double_buffer_trace.json"
OUT_EVENTS_HEX = OUT_DIR / "double_buffer_events.hex"
OUT_BUFFERS_HEX = OUT_DIR / "double_buffer_final_state.hex"

TILE_BYTES = 256
TILES = 6
LOAD_LATENCY = 4
COMPUTE_LATENCY = 7
WRITE_LATENCY = 3


def tile_checksum(tile_id: int) -> int:
    return (0x13579BDF + tile_id * 0x1020304) & 0xFFFFFFFF


def write_hex(path: Path, values: list[int], width: int) -> None:
    mask = (1 << (width * 4)) - 1
    path.write_text("\n".join(f"{value & mask:0{width}x}" for value in values) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    buffers = [
        {"valid": False, "tile": None, "checksum": 0},
        {"valid": False, "tile": None, "checksum": 0},
    ]
    buffer_free = [0, 0]
    load_free = 0
    compute_free = 0
    write_free = 0
    events: list[dict] = []
    hazards: list[dict] = []

    for tile in range(TILES):
        load_buffer = tile % 2
        compute_buffer = load_buffer
        checksum = tile_checksum(tile)

        load_start = max(load_free, buffer_free[load_buffer])
        load_done = load_start + LOAD_LATENCY
        if buffers[load_buffer]["valid"] and load_start < buffer_free[load_buffer]:
            hazards.append(
                {
                    "tile": tile,
                    "buffer": load_buffer,
                    "type": "overwrite_valid_buffer",
                    "load_start": load_start,
                    "buffer_free": buffer_free[load_buffer],
                }
            )

        buffers[load_buffer] = {"valid": True, "tile": tile, "checksum": checksum}
        events.append(
            {
                "tile": tile,
                "stage": "load",
                "buffer": load_buffer,
                "start": load_start,
                "done": load_done,
                "checksum": checksum,
            }
        )
        load_free = load_start + TILE_BYTES // 64

        compute_start = max(compute_free, load_done)
        compute_done = compute_start + COMPUTE_LATENCY
        if not buffers[compute_buffer]["valid"] or buffers[compute_buffer]["tile"] != tile:
            hazards.append({"tile": tile, "buffer": compute_buffer, "type": "compute_without_loaded_tile"})

        events.append(
            {
                "tile": tile,
                "stage": "compute",
                "buffer": compute_buffer,
                "start": compute_start,
                "done": compute_done,
                "checksum": checksum,
            }
        )
        compute_free = compute_done

        write_start = max(write_free, compute_done)
        write_done = write_start + WRITE_LATENCY
        events.append(
            {
                "tile": tile,
                "stage": "write",
                "buffer": compute_buffer,
                "start": write_start,
                "done": write_done,
                "checksum": checksum,
            }
        )
        write_free = write_done
        buffer_free[compute_buffer] = write_done
        buffers[compute_buffer]["valid"] = True

    if hazards:
        raise AssertionError(f"double buffer schedule has hazards: {hazards}")

    end_cycle = write_free
    for idx, buffer in enumerate(buffers):
        if buffer_free[idx] <= end_cycle:
            buffer["valid"] = False
    stage_ids = {"load": 0, "compute": 1, "write": 2}
    events.sort(key=lambda event: (event["start"], stage_ids[event["stage"]], event["tile"]))

    OUT_JSON.write_text(
        json.dumps(
            {
                "tile_bytes": TILE_BYTES,
                "tiles": TILES,
                "load_latency": LOAD_LATENCY,
                "compute_latency": COMPUTE_LATENCY,
                "write_latency": WRITE_LATENCY,
                "end_cycle": end_cycle,
                "events": events,
                "final_buffers": buffers,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_hex(
        OUT_EVENTS_HEX,
        [
            ((event["tile"] & 0xFF) << 56)
            | ((stage_ids[event["stage"]] & 0xF) << 52)
            | ((event["buffer"] & 0xF) << 48)
            | ((event["start"] & 0xFFFF) << 32)
            | ((event["done"] & 0xFFFF) << 16)
            | (event["checksum"] & 0xFFFF)
            for event in events
        ],
        16,
    )
    write_hex(
        OUT_BUFFERS_HEX,
        [
            ((idx & 0xF) << 36)
            | ((int(buffer["valid"]) & 0x1) << 32)
            | (buffer["checksum"] & 0xFFFFFFFF)
            for idx, buffer in enumerate(buffers)
        ],
        10,
    )

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_EVENTS_HEX}")
    print(f"Wrote {OUT_BUFFERS_HEX}")


if __name__ == "__main__":
    main()
