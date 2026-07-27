from __future__ import annotations

from dataclasses import dataclass

from .compiler import ModelPackage
from .dtypes import DType, OpType


DEFAULT_ARENA_BYTES = 64 * 1024 * 1024
MAX_FIRMWARE_BUFFERS = 128
ALLOCATION_ALIGNMENT = 64

PL_OPERATORS = frozenset(
    {
        OpType.GEMM,
        OpType.CONV2D_GEMM,
        OpType.ACTIVATION,
        OpType.REQUANTIZE,
        OpType.POOL,
        OpType.UPSAMPLE,
        OpType.CONCAT,
        OpType.RESHAPE,
    }
)


@dataclass(frozen=True)
class DeploymentPreflight:
    layers: int
    buffers: int
    descriptor_bytes: int
    arena_bytes_required: int
    arena_bytes_available: int


def _align_up(value: int, alignment: int = ALLOCATION_ALIGNMENT) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _buffer(package: ModelPackage, index: int, label: str):
    if index < 0 or index >= len(package.buffers):
        raise ValueError(f"{label} references invalid buffer index {index}")
    return package.buffers[index]


def _require_int4_pair(package: ModelPackage, layer_index: int, input_index: int, output_index: int) -> None:
    source = _buffer(package, input_index, f"layer {layer_index} input")
    output = _buffer(package, output_index, f"layer {layer_index} output")
    if source.dtype != DType.INT4 or output.dtype != DType.INT4:
        raise ValueError(
            f"layer {layer_index} requires packed INT4 input/output, got "
            f"{source.dtype.name}/{output.dtype.name}"
        )


def validate_model_package(
    package: ModelPackage,
    *,
    arena_bytes: int = DEFAULT_ARENA_BYTES,
) -> DeploymentPreflight:
    """Reject packages that the current PYNQ-Z2 daemon/INT4 PL cannot execute."""
    if not isinstance(package, ModelPackage):
        raise TypeError("package must be a ModelPackage")
    if arena_bytes <= 0:
        raise ValueError("arena_bytes must be positive")
    if not package.layers:
        raise ValueError("model package has no executable layers")
    if not package.buffers:
        raise ValueError("model package has no buffers")
    if len(package.buffers) + 1 > MAX_FIRMWARE_BUFFERS:
        raise ValueError(
            f"model needs {len(package.buffers) + 1} firmware handles including "
            f"its descriptor; limit is {MAX_FIRMWARE_BUFFERS}"
        )

    _buffer(package, package.input_index, "model input")
    _buffer(package, package.output_index, "model output")
    if package.buffers[package.input_index].dtype != DType.INT4:
        raise ValueError("current board image requires a packed INT4 model input")
    if package.buffers[package.output_index].dtype != DType.INT4:
        raise ValueError("current board image requires a packed INT4 model output")

    for index, layer in enumerate(package.layers):
        if layer.op_type not in PL_OPERATORS:
            raise ValueError(
                f"layer {index} uses {layer.op_type.name}, which is not in the "
                "current PL execution path"
            )
        source = _buffer(package, layer.input0, f"layer {index} input")
        output = _buffer(package, layer.output, f"layer {index} output")

        if layer.op_type == OpType.CONV2D_GEMM:
            weight = _buffer(package, layer.weight, f"layer {index} weight")
            if source.dtype != DType.INT4 or weight.dtype != DType.INT4 or output.dtype != DType.INT32:
                raise ValueError(f"layer {index} Conv2D requires INT4/INT4 -> INT32")
            if layer.bias:
                bias = _buffer(package, layer.bias, f"layer {index} bias")
                if bias.dtype != DType.INT32:
                    raise ValueError(f"layer {index} Conv2D bias must be INT32")
            if (
                layer.n != 1
                or layer.groups != 1
                or not 1 <= layer.kernel_h <= 16
                or not 1 <= layer.kernel_w <= 16
                or not 1 <= layer.stride_h <= 255
                or not 1 <= layer.stride_w <= 255
                or not 0 <= layer.pad_h <= 255
                or not 0 <= layer.pad_w <= 255
            ):
                raise ValueError(f"layer {index} Conv2D geometry is outside the current PL limits")
        elif layer.op_type == OpType.GEMM:
            weight = _buffer(package, layer.weight, f"layer {index} weight")
            if source.dtype != DType.INT4 or weight.dtype != DType.INT4 or output.dtype != DType.INT32:
                raise ValueError(f"layer {index} GEMM requires INT4/INT4 -> INT32")
            if layer.n <= 0 or layer.c <= 0 or layer.out_c <= 0:
                raise ValueError(f"layer {index} GEMM dimensions must be positive")
            if layer.c % 2 and layer.n > 1:
                raise ValueError(
                    f"layer {index} GEMM has odd K with multiple A rows; "
                    "the packed row ABI requires an even K"
                )
            if layer.out_c % 2 and layer.c > 1:
                raise ValueError(
                    f"layer {index} GEMM has odd N with multiple B rows; "
                    "lower the classifier through 1x1 Conv2D or pad N"
                )
        elif layer.op_type == OpType.REQUANTIZE:
            scale = _buffer(package, layer.scale, f"layer {index} scale")
            if source.dtype != DType.INT32 or output.dtype != DType.INT4 or scale.dtype != DType.INT32:
                raise ValueError(f"layer {index} requantize requires INT32 -> INT4 with INT32 parameters")
        elif layer.op_type in (
            OpType.ACTIVATION,
            OpType.POOL,
            OpType.UPSAMPLE,
            OpType.RESHAPE,
        ):
            _require_int4_pair(package, index, layer.input0, layer.output)
            if layer.op_type == OpType.RESHAPE and source.size_bytes != output.size_bytes:
                raise ValueError(f"layer {index} reshape changes packed storage size")
        elif layer.op_type == OpType.CONCAT:
            _require_int4_pair(package, index, layer.input0, layer.output)
            _require_int4_pair(package, index, layer.input1, layer.output)
            if layer.input2:
                _require_int4_pair(package, index, layer.input2, layer.output)

        dimensions = (
            layer.n,
            layer.c,
            layer.h,
            layer.w,
            layer.out_c,
            layer.out_h,
            layer.out_w,
        )
        if any(value < 0 or value > 0xFFFF for value in dimensions):
            raise ValueError(f"layer {index} has a dimension outside the 16-bit PL register range")

    descriptor_bytes = len(package.serialize_descriptor())
    offset = 0
    for buffer in package.buffers:
        offset = _align_up(offset)
        offset += buffer.size_bytes
    offset = _align_up(offset)
    offset += descriptor_bytes
    if offset > arena_bytes:
        raise ValueError(
            f"model needs {offset} bytes in the firmware arena, but only "
            f"{arena_bytes} bytes are configured"
        )
    return DeploymentPreflight(
        layers=len(package.layers),
        buffers=len(package.buffers),
        descriptor_bytes=descriptor_bytes,
        arena_bytes_required=offset,
        arena_bytes_available=arena_bytes,
    )
