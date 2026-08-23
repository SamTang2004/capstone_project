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


def unpack_transformer_activations_int8(raw: bytes, m: int, k: int) -> list[list[int]]:
    """Decode [M-tile][K][8 row lanes] activations into row-major MxK."""
    if m <= 0 or k <= 0:
        raise ValueError("Activation dimensions must be positive")
    expected = ((m + 7) // 8) * k * 8
    raw = bytes(raw)
    if len(raw) != expected:
        raise ValueError(f"Packed activation has {len(raw)} bytes, expected {expected}")
    values = [[0] * k for _ in range(m)]
    cursor = 0
    for m0 in range(0, m, 8):
        for inner in range(k):
            for lane in range(8):
                value = raw[cursor]
                cursor += 1
                value = value if value < 128 else value - 256
                if m0 + lane < m:
                    values[m0 + lane][inner] = value
                # Output-producing kernels may leave padding lanes unspecified.
    return values


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


def unpack_transformer_weights(raw: bytes, k: int, n: int, *, int4: bool = False) -> list[list[int]]:
    """Decode tiled W8/W4 accelerator weights into a row-major KxN matrix."""
    if k <= 0 or n <= 0:
        raise ValueError("Weight dimensions must be positive")
    expected = ((n + 7) // 8) * k * (4 if int4 else 8)
    raw = bytes(raw)
    if len(raw) != expected:
        raise ValueError(f"Packed weight has {len(raw)} bytes, expected {expected}")
    values = [[0] * n for _ in range(k)]
    cursor = 0
    for n0 in range(0, n, 8):
        for inner in range(k):
            if int4:
                for pair in range(4):
                    packed = raw[cursor]
                    cursor += 1
                    for half in range(2):
                        column = n0 + pair * 2 + half
                        nibble = (packed >> (4 * half)) & 0xF
                        value = nibble if nibble < 8 else nibble - 16
                        if column < n:
                            values[inner][column] = value
                        elif value:
                            raise ValueError("Non-zero padded W4 lane")
            else:
                for lane in range(8):
                    raw_value = raw[cursor]
                    cursor += 1
                    column = n0 + lane
                    value = raw_value if raw_value < 128 else raw_value - 256
                    if column < n:
                        values[inner][column] = value
                    elif value:
                        raise ValueError("Non-zero padded W8 lane")
    return values
