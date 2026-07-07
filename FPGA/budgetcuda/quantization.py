from __future__ import annotations


def quantize_symmetric_int8(values, scale: float | None = None):
    flat = _flatten(values)
    if scale is None:
        max_abs = max((abs(float(value)) for value in flat), default=0.0)
        scale = 1.0 if max_abs == 0.0 else max_abs / 127.0
    q = [max(-128, min(127, int(round(float(value) / float(scale))))) for value in flat]
    return _maybe_numpy_array(q, "int8"), float(scale)


def dequantize_symmetric_int8(values, scale: float):
    out = [int(value) * float(scale) for value in _flatten(values)]
    return _maybe_numpy_array(out, "float32")


def _flatten(values) -> list:
    if hasattr(values, "reshape") and hasattr(values, "tolist"):
        values = values.reshape(-1).tolist()
    out = []

    def visit(item):
        if isinstance(item, (list, tuple)):
            for subitem in item:
                visit(subitem)
        else:
            out.append(item)

    visit(values)
    return out


def _maybe_numpy_array(values, dtype):
    try:
        import numpy as np

        return np.asarray(values, dtype=dtype)
    except ModuleNotFoundError:
        return values
