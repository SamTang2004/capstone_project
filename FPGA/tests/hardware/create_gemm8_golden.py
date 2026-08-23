"""Generate deterministic W8A8 and W4A8 vectors for the 8x8 tile core."""

from __future__ import annotations

import json
import random
from pathlib import Path


OUT = Path(__file__).resolve().parent / "golden_traces"


def pack(values: list[int], bits: int) -> int:
    mask = (1 << bits) - 1
    return sum((value & mask) << (lane * bits) for lane, value in enumerate(values))


def write_case(name: str, *, weight_bits: int, rows: int, cols: int, k: int, seed: int) -> None:
    rng = random.Random(seed)
    activations = [[rng.randint(-128, 127) for _ in range(k)] for _ in range(8)]
    weight_limit = 7 if weight_bits == 4 else 127
    weight_min = -8 if weight_bits == 4 else -128
    weights = [[rng.randint(weight_min, weight_limit) for _ in range(8)] for _ in range(k)]
    expected = [[0 for _ in range(8)] for _ in range(8)]
    streams: list[int] = []
    for reduction in range(k):
        a_values = [activations[row][reduction] for row in range(8)]
        w_values = weights[reduction]
        a_word = pack(a_values, 8)
        w_word = pack(w_values, weight_bits)
        streams.append((a_word << 64) | w_word)
        for row in range(rows):
            for col in range(cols):
                expected[row][col] += a_values[row] * w_values[col]

    (OUT / f"{name}_inputs.hex").write_text(
        "".join(f"{word:032x}\n" for word in streams), encoding="ascii"
    )
    flattened = [expected[row][col] for row in range(8) for col in range(8)]
    (OUT / f"{name}_expected.hex").write_text(
        "".join(f"{value & 0xFFFFFFFF:08x}\n" for value in flattened), encoding="ascii"
    )
    (OUT / f"{name}.json").write_text(
        json.dumps({
            "weight_bits": weight_bits,
            "rows": rows,
            "cols": cols,
            "k": k,
            "activations": activations,
            "weights": weights,
            "expected": expected,
        }, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_case("gemm8_w8", weight_bits=8, rows=8, cols=8, k=11, seed=81)
    write_case("gemm8_w4", weight_bits=4, rows=5, cols=6, k=13, seed=48)
    print(f"Wrote 8x8 GEMM vectors to {OUT}")


if __name__ == "__main__":
    main()
