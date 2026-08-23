from __future__ import annotations


def _checked_int(value: object, low: int, high: int) -> int:
    value = int(value)
    if value < low or value > high:
        raise ValueError(f"Quantized value {value} is outside [{low}, {high}]")
    return value


def pack_transformer_activations_int8(values, m: int, k: int) -> bytes:
    """Pack row-major MxK activations as [M-tile][K][8 row lanes]."""
    if m <= 0 or k <= 0 or len(values) != m or any(len(row) != k for row in values):
        raise ValueError("Activation values must have shape MxK")
    packed = bytearray(((m + 7) // 8) * k * 8)
    cursor = 0
    for m0 in range(0, m, 8):
        for inner in range(k):
            for lane in range(8):
                packed[cursor] = (
                    _checked_int(values[m0 + lane][inner], -128, 127) & 0xFF
                    if m0 + lane < m else 0
                )
                cursor += 1
    return bytes(packed)


def pack_transformer_weights(values, k: int, n: int, *, int4: bool = False) -> bytes:
    """Pack row-major KxN weights as [N-tile][K][8 column lanes]."""
    if k <= 0 or n <= 0 or len(values) != k or any(len(row) != n for row in values):
        raise ValueError("Weight values must have shape KxN")
    packed = bytearray()
    for n0 in range(0, n, 8):
        for inner in range(k):
            lanes = [
                _checked_int(values[inner][n0 + lane], -8 if int4 else -128, 7 if int4 else 127)
                if n0 + lane < n else 0
                for lane in range(8)
            ]
            if int4:
                packed.extend((lanes[lane] & 0xF) | ((lanes[lane + 1] & 0xF) << 4) for lane in range(0, 8, 2))
            else:
                packed.extend(value & 0xFF for value in lanes)
    return bytes(packed)
