from __future__ import annotations

from enum import IntEnum


BCUDA_MAGIC = 0x42435544
BCUDA_VERSION = 1


class DType(IntEnum):
    INT8 = 1
    INT16 = 2
    INT32 = 3
    INT4 = 4


class Layout(IntEnum):
    ROW_MAJOR = 1
    NCHW = 2
    NHWC = 3


class BufferUsage(IntEnum):
    WEIGHT = 1
    ACTIVATION = 2
    WORKSPACE = 3
    OUTPUT = 4
    DESCRIPTOR = 5


class Command(IntEnum):
    ALLOC = 1
    WRITE = 2
    READ = 3
    EXECUTE = 4
    FREE = 5


class Status(IntEnum):
    OK = 0
    GENERAL_ERROR = 1
    INVALID_COMMAND = 2
    INVALID_HANDLE = 3
    OUT_OF_MEMORY = 4
    SHAPE_MISMATCH = 5
    UNSUPPORTED_DTYPE = 6
    EXECUTION_FAILED = 7


class OpType(IntEnum):
    GEMM = 1
    GRAPH = 2
    CONV2D_GEMM = 3
    ACTIVATION = 4
    POOL = 5
    UPSAMPLE = 6
    CONCAT = 7
    RESHAPE = 8
    DETECT_DECODE = 9
    NMS = 10
    REQUANTIZE = 11


DTYPE_ALIASES = {
    "int4": DType.INT4,
    "qint4": DType.INT4,
    "int8": DType.INT8,
    "int16": DType.INT16,
    "int32": DType.INT32,
}

LAYOUT_ALIASES = {
    "row_major": Layout.ROW_MAJOR,
    "nchw": Layout.NCHW,
    "nhwc": Layout.NHWC,
}

USAGE_ALIASES = {
    "weight": BufferUsage.WEIGHT,
    "activation": BufferUsage.ACTIVATION,
    "workspace": BufferUsage.WORKSPACE,
    "output": BufferUsage.OUTPUT,
    "descriptor": BufferUsage.DESCRIPTOR,
}

DTYPE_SIZES = {
    DType.INT8: 1,
    DType.INT16: 2,
    DType.INT32: 4,
}

DTYPE_BITS = {
    DType.INT4: 4,
    DType.INT8: 8,
    DType.INT16: 16,
    DType.INT32: 32,
}


def normalize_dtype(dtype: str | int | DType) -> DType:
    if isinstance(dtype, DType):
        return dtype
    if isinstance(dtype, str):
        try:
            return DTYPE_ALIASES[dtype.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported dtype: {dtype}") from exc
    return DType(dtype)


def normalize_layout(layout: str | int | Layout) -> Layout:
    if isinstance(layout, Layout):
        return layout
    if isinstance(layout, str):
        try:
            return LAYOUT_ALIASES[layout.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported layout: {layout}") from exc
    return Layout(layout)


def normalize_usage(usage: str | int | BufferUsage) -> BufferUsage:
    if isinstance(usage, BufferUsage):
        return usage
    if isinstance(usage, str):
        try:
            return USAGE_ALIASES[usage.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported buffer usage: {usage}") from exc
    return BufferUsage(usage)


def dtype_size(dtype: str | int | DType) -> int:
    dtype = normalize_dtype(dtype)
    if dtype == DType.INT4:
        raise ValueError("INT4 is packed; use dtype_nbytes(dtype, element_count)")
    return DTYPE_SIZES[dtype]


def dtype_bits(dtype: str | int | DType) -> int:
    return DTYPE_BITS[normalize_dtype(dtype)]


def dtype_nbytes(dtype: str | int | DType, element_count: int) -> int:
    element_count = int(element_count)
    if element_count < 0:
        raise ValueError("Element count must be non-negative")
    return (dtype_bits(dtype) * element_count + 7) // 8
