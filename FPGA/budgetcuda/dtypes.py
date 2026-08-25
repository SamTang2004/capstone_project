from __future__ import annotations

from enum import IntEnum


BCUDA_MAGIC = 0x42435544
BCUDA_VERSION = 1


class DType(IntEnum):
    INT8 = 1
    INT16 = 2
    INT32 = 3


class Layout(IntEnum):
    ROW_MAJOR = 1
    NCHW = 2
    NHWC = 3
    CONV_WEIGHT_TILED_INT8 = 4
    TRANSFORMER_ACTIVATION_TILED_INT8 = 5
    TRANSFORMER_WEIGHT_TILED_INT8 = 6
    TRANSFORMER_WEIGHT_TILED_INT4 = 7
    TRANSFORMER_DYNAMIC_B_TILED_INT8 = 8


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
    LAYERNORM = 12
    GELU = 13
    SOFTMAX = 14
    RESIDUAL_ADD = 15
    ACTIVATION_B_PACK = 16
    TOKEN_CONCAT = 17
    TOKEN_EMBEDDING = 18
    TOKEN_PRUNE = 19


DTYPE_ALIASES = {
    "int8": DType.INT8,
    "int16": DType.INT16,
    "int32": DType.INT32,
}

LAYOUT_ALIASES = {
    "row_major": Layout.ROW_MAJOR,
    "nchw": Layout.NCHW,
    "nhwc": Layout.NHWC,
    "conv_weight_tiled_int8": Layout.CONV_WEIGHT_TILED_INT8,
    "transformer_activation_tiled_int8": Layout.TRANSFORMER_ACTIVATION_TILED_INT8,
    "transformer_weight_tiled_int8": Layout.TRANSFORMER_WEIGHT_TILED_INT8,
    "transformer_weight_tiled_int4": Layout.TRANSFORMER_WEIGHT_TILED_INT4,
    "transformer_dynamic_b_tiled_int8": Layout.TRANSFORMER_DYNAMIC_B_TILED_INT8,
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
    return DTYPE_SIZES[normalize_dtype(dtype)]
