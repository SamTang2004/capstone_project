from __future__ import annotations


def pack_int4(values) -> bytes:
    """Pack signed INT4 values with the even element in the low nibble."""

    flat = _flatten(values)
    packed = bytearray((len(flat) + 1) // 2)
    for index, value in enumerate(flat):
        value = int(value)
        if value < -8 or value > 7:
            raise ValueError(f"INT4 value out of range at element {index}: {value}")
        nibble = value & 0xF
        if index & 1:
            packed[index // 2] |= nibble << 4
        else:
            packed[index // 2] = nibble
    return bytes(packed)


def unpack_int4(data: bytes | bytearray | memoryview, element_count: int) -> list[int]:
    """Unpack signed INT4 values from the BudgetCUDA packed tensor ABI."""

    raw = bytes(data)
    element_count = int(element_count)
    if element_count < 0:
        raise ValueError("Element count must be non-negative")
    if len(raw) * 2 < element_count:
        raise ValueError("Packed INT4 data is shorter than the requested element count")
    values = []
    for index in range(element_count):
        byte = raw[index // 2]
        nibble = (byte >> (4 if index & 1 else 0)) & 0xF
        values.append(nibble - 16 if nibble & 0x8 else nibble)
    return values


def _flatten(values) -> list:
    if hasattr(values, "reshape") and hasattr(values, "tolist"):
        return values.reshape(-1).tolist()
    output = []

    def visit(item):
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            output.append(item)

    visit(values)
    return output
