"""Generate the deterministic golden trace for the dual-INT4 row streamer."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "golden_traces" / "int4_streaming_rows_trace.json"


def packed_word(stream: int, row: int) -> int:
    word = 0
    for lane in range(8):
        value = (stream * 5 + row * 3 + lane) & 0xF
        word |= value << (lane * 4)
    return word


def main() -> None:
    # Deliberately interleave the producer streams and delay B1 row 0. This
    # checks that the joiner holds A/B0 for each customer without reordering.
    arrivals: list[dict[str, int | str]] = []
    for row in range(8):
        arrivals.append(
            {"stream": "A", "row": row, "mask": 0xFF, "data": packed_word(0, row)}
        )
        arrivals.append(
            {"stream": "B0", "row": row, "mask": 0xFF, "data": packed_word(1, row)}
        )
        if row:
            arrivals.append(
                {
                    "stream": "B1",
                    "row": row,
                    "mask": 0xFF,
                    "data": packed_word(2, row),
                }
            )
    arrivals.append(
        {"stream": "B1", "row": 0, "mask": 0xFF, "data": packed_word(2, 0)}
    )

    outputs = [
        {
            "row": row,
            "a_mask": 0xFF,
            "a_data": packed_word(0, row),
            "b0_mask": 0xFF,
            "b0_data": packed_word(1, row),
            "b1_mask": 0xFF,
            "b1_data": packed_word(2, row),
        }
        for row in range(8)
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "active_arrays": 2,
                "a_row_mask": 0xFF,
                "b_row_mask": 0xFF,
                "arrivals": arrivals,
                "outputs": outputs,
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
