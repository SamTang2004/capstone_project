from __future__ import annotations

from dataclasses import dataclass

from .dtypes import DType, Layout, dtype_bits, dtype_size, normalize_dtype, normalize_layout


@dataclass(frozen=True)
class Tensor:
    buffer_id: int
    shape: tuple[int, ...]
    dtype: DType
    layout: Layout
    size_bytes: int
    device: object

    def __init__(self, buffer_id, shape, dtype, layout, size_bytes, device):
        object.__setattr__(self, "buffer_id", int(buffer_id))
        object.__setattr__(self, "shape", tuple(int(dim) for dim in shape))
        object.__setattr__(self, "dtype", normalize_dtype(dtype))
        object.__setattr__(self, "layout", normalize_layout(layout))
        object.__setattr__(self, "size_bytes", int(size_bytes))
        object.__setattr__(self, "device", device)

    @property
    def nbytes(self) -> int:
        return self.size_bytes

    @property
    def itemsize(self) -> int:
        return dtype_size(self.dtype)

    @property
    def bits_per_element(self) -> int:
        return dtype_bits(self.dtype)

    @property
    def ndim(self) -> int:
        return len(self.shape)
