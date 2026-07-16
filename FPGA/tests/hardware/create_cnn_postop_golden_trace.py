"""Golden vectors for the next quantized CNN post-processing data paths.

The PL contract is deliberately explicit: QADD consumes two tensors in the
same INT8 quantization domain, average pooling rounds toward zero, and SiLU
uses a resident 256-entry INT8 LUT selected by the host/compiler.
"""

from __future__ import annotations

from pathlib import Path


OUT = Path(__file__).resolve().parent / "golden_traces"


def sat_i8(value: int) -> int:
    return max(-128, min(127, value))


def qadd(a: list[int], b: list[int], zero_point: int = 0) -> list[int]:
    return [sat_i8(x + y - zero_point) for x, y in zip(a, b)]


def avg_pool(values: list[int], h: int, w: int, kernel: int, stride: int) -> list[int]:
    out = []
    for oy in range((h - kernel) // stride + 1):
        for ox in range((w - kernel) // stride + 1):
            window = [values[(oy * stride + ky) * w + ox * stride + kx] for ky in range(kernel) for kx in range(kernel)]
            out.append(int(sum(window) / len(window)))  # C/RTL signed division: toward zero.
    return out


def silu_lut(input_scale: float, output_scale: float) -> list[int]:
    import math

    return [sat_i8(round(((q * input_scale) / (1.0 + math.exp(-(q * input_scale)))) / output_scale)) for q in range(-128, 128)]


def write_hex(name: str, values: list[int]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(f"{value & 0xff:02x}" for value in values) + "\n", encoding="ascii")


def main() -> None:
    a = [-128, -10, -1, 0, 4, 100, 120, 127]
    b = [-1, 5, -3, 0, 4, 40, 20, 1]
    write_hex("qadd_a.hex", a)
    write_hex("qadd_b.hex", b)
    write_hex("qadd_expected.hex", qadd(a, b))
    write_hex("avgpool_input.hex", [-7, -2, 3, 8, 4, -5, 6, 9, -1])
    write_hex("avgpool_expected.hex", avg_pool([-7, -2, 3, 8, 4, -5, 6, 9, -1], 3, 3, 2, 1))
    write_hex("silu_lut.hex", silu_lut(1.0 / 16.0, 1.0 / 16.0))
    write_hex("silu_input.hex", [-128, -32, -1, 0, 16, 64, 127])
    lut = silu_lut(1.0 / 16.0, 1.0 / 16.0)
    write_hex("silu_expected.hex", [lut[value + 128] for value in [-128, -32, -1, 0, 16, 64, 127]])
    print("CNN postop golden traces written")


if __name__ == "__main__":
    main()
