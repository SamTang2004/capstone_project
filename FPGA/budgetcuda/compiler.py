from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Iterable

from .dtypes import BufferUsage, DType, Layout, OpType, dtype_size, normalize_dtype, normalize_layout
from .model import Model


MODEL_MAGIC = 0x444D4342
MODEL_VERSION = 1
MODEL_DESC_STRUCT = struct.Struct("<IIIIIIIIIIIIII")
LAYER_DESC_STRUCT = struct.Struct("<IIIIIIIIIIIIIIIIIIIIIIII")
BUFFER_DESC_STRUCT = struct.Struct("<IIIIIIII")


@dataclass(frozen=True)
class HostBuffer:
    name: str
    shape: tuple[int, ...]
    dtype: DType
    layout: Layout = Layout.ROW_MAJOR
    usage: BufferUsage = BufferUsage.ACTIVATION
    data: bytes | None = None

    @classmethod
    def from_values(
        cls,
        name: str,
        shape: Iterable[int],
        dtype: str | int | DType,
        values,
        usage: BufferUsage = BufferUsage.WEIGHT,
        layout: str | int | Layout = Layout.ROW_MAJOR,
    ) -> "HostBuffer":
        dtype = normalize_dtype(dtype)
        layout = normalize_layout(layout)
        itemsize = dtype_size(dtype)
        flat = _flatten(values)
        raw = b"".join(int(value).to_bytes(itemsize, "little", signed=True) for value in flat)
        return cls(name, tuple(shape), dtype, layout, usage, raw)

    @property
    def size_bytes(self) -> int:
        return _num_elements(self.shape) * dtype_size(self.dtype)


@dataclass(frozen=True)
class Layer:
    op_type: OpType
    input0: int = 0
    input1: int = 0
    input2: int = 0
    output: int = 0
    weight: int = 0
    bias: int = 0
    scale: int = 0
    n: int = 0
    c: int = 0
    h: int = 0
    w: int = 0
    out_c: int = 0
    out_h: int = 0
    out_w: int = 0
    kernel_h: int = 0
    kernel_w: int = 0
    stride_h: int = 0
    stride_w: int = 0
    pad_h: int = 0
    pad_w: int = 0
    groups: int = 1
    flags: int = 0

    def pack(self) -> bytes:
        return LAYER_DESC_STRUCT.pack(
            int(self.op_type),
            self.input0,
            self.input1,
            self.input2,
            self.output,
            self.weight,
            self.bias,
            self.scale,
            self.n,
            self.c,
            self.h,
            self.w,
            self.out_c,
            self.out_h,
            self.out_w,
            self.kernel_h,
            self.kernel_w,
            self.stride_h,
            self.stride_w,
            self.pad_h,
            self.pad_w,
            self.groups,
            self.flags,
            0,
        )


@dataclass
class ModelPackage:
    buffers: list[HostBuffer]
    layers: list[Layer]
    input_index: int
    output_index: int
    activation_arena_a: int = 0
    activation_arena_b: int = 0
    workspace_buffer: int = 0
    flags: int = 0
    name_to_index: dict[str, int] = field(init=False)

    def __post_init__(self):
        self.name_to_index = {buffer.name: idx for idx, buffer in enumerate(self.buffers)}

    def serialize_descriptor(self, device_buffer_ids: list[int] | None = None) -> bytes:
        buffer_ids = device_buffer_ids or [0] * len(self.buffers)
        buffer_table_offset = MODEL_DESC_STRUCT.size
        layer_desc_offset = buffer_table_offset + len(self.buffers) * BUFFER_DESC_STRUCT.size
        model = MODEL_DESC_STRUCT.pack(
            MODEL_MAGIC,
            MODEL_VERSION,
            len(self.layers),
            layer_desc_offset,
            len(self.buffers),
            buffer_table_offset,
            self.input_index,
            self.output_index,
            self.activation_arena_a,
            self.activation_arena_b,
            self.workspace_buffer,
            self.flags,
            0,
            0,
        )
        buffer_table = b"".join(
            BUFFER_DESC_STRUCT.pack(
                buffer_ids[idx],
                int(buffer.dtype),
                int(buffer.layout),
                int(buffer.usage),
                buffer.size_bytes,
                len(buffer.shape),
                buffer.shape[0] if len(buffer.shape) > 0 else 1,
                buffer.shape[1] if len(buffer.shape) > 1 else 1,
            )
            for idx, buffer in enumerate(self.buffers)
        )
        layers = b"".join(layer.pack() for layer in self.layers)
        return model + buffer_table + layers


class GraphBuilder:
    def __init__(self, name: str = "graph"):
        self.name = name
        self.buffers: list[HostBuffer] = []
        self.layers: list[Layer] = []

    def add_buffer(
        self,
        name: str,
        shape: Iterable[int],
        dtype: str | int | DType,
        usage: BufferUsage = BufferUsage.ACTIVATION,
        data=None,
        layout: str | int | Layout = Layout.ROW_MAJOR,
    ) -> int:
        full_name = name if name.startswith(f"{self.name}.") else f"{self.name}.{name}"
        if data is None:
            buffer = HostBuffer(full_name, tuple(shape), normalize_dtype(dtype), normalize_layout(layout), usage)
        else:
            buffer = HostBuffer.from_values(full_name, tuple(shape), dtype, data, usage=usage, layout=layout)
        self.buffers.append(buffer)
        return len(self.buffers) - 1

    def conv2d(
        self,
        input_idx: int,
        weight_idx: int,
        output_idx: int,
        *,
        n: int,
        c: int,
        h: int,
        w: int,
        out_c: int,
        out_h: int,
        out_w: int,
        kernel_h: int,
        kernel_w: int,
        stride_h: int = 1,
        stride_w: int = 1,
        pad_h: int = 0,
        pad_w: int = 0,
        bias_idx: int = 0,
    ) -> None:
        self.layers.append(
            Layer(
                OpType.CONV2D_GEMM,
                input0=input_idx,
                output=output_idx,
                weight=weight_idx,
                bias=bias_idx,
                n=n,
                c=c,
                h=h,
                w=w,
                out_c=out_c,
                out_h=out_h,
                out_w=out_w,
                kernel_h=kernel_h,
                kernel_w=kernel_w,
                stride_h=stride_h,
                stride_w=stride_w,
                pad_h=pad_h,
                pad_w=pad_w,
            )
        )

    def relu(self, input_idx: int, output_idx: int) -> None:
        self.layers.append(Layer(OpType.ACTIVATION, input0=input_idx, output=output_idx))

    def leaky_relu(self, input_idx: int, output_idx: int, right_shift: int) -> None:
        if right_shift < 0 or right_shift > 7:
            raise ValueError("Leaky ReLU right_shift must be in 0..7")
        self.layers.append(Layer(OpType.ACTIVATION, input0=input_idx, output=output_idx, flags=1 | (right_shift << 8)))

    def requantize_params(self, name: str, multiplier: int, right_shift: int, output_zero_point: int = 0, input_zero_point: int = 0) -> int:
        if right_shift < 0 or right_shift > 31:
            raise ValueError("right_shift must be in 0..31")
        return self.add_buffer(name, (4,), DType.INT32, usage=BufferUsage.WEIGHT, data=[multiplier, right_shift, input_zero_point, output_zero_point])

    def requantize(self, input_idx: int, output_idx: int, scale_idx: int) -> None:
        self.layers.append(Layer(OpType.REQUANTIZE, input0=input_idx, output=output_idx, scale=scale_idx))

    def maxpool2d(
        self,
        input_idx: int,
        output_idx: int,
        *,
        n: int,
        c: int,
        h: int,
        w: int,
        out_h: int,
        out_w: int,
        kernel_h: int,
        kernel_w: int,
        stride_h: int,
        stride_w: int,
        pad_h: int = 0,
        pad_w: int = 0,
    ) -> None:
        self.layers.append(
            Layer(
                OpType.POOL,
                input0=input_idx,
                output=output_idx,
                n=n,
                c=c,
                h=h,
                w=w,
                out_h=out_h,
                out_w=out_w,
                kernel_h=kernel_h,
                kernel_w=kernel_w,
                stride_h=stride_h,
                stride_w=stride_w,
                pad_h=pad_h,
                pad_w=pad_w,
            )
        )

    def upsample(self, input_idx: int, output_idx: int, *, n: int, c: int, h: int, w: int, out_h: int, out_w: int) -> None:
        self.layers.append(Layer(OpType.UPSAMPLE, input0=input_idx, output=output_idx, n=n, c=c, h=h, w=w, out_h=out_h, out_w=out_w))

    def concat_channels(self, input0: int, input1: int, output_idx: int, *, n: int, h: int, w: int, out_c: int, input2: int = 0) -> None:
        self.layers.append(Layer(OpType.CONCAT, input0=input0, input1=input1, input2=input2, output=output_idx, n=n, h=h, w=w, out_c=out_c))

    def reshape(self, input_idx: int, output_idx: int) -> None:
        self.layers.append(Layer(OpType.RESHAPE, input0=input_idx, output=output_idx))

    def package(self, input_idx: int, output_idx: int) -> ModelPackage:
        return ModelPackage(self.buffers, self.layers, input_idx, output_idx)


def upload_model(device, package: ModelPackage) -> Model:
    tensors = []
    descriptor_tensor = None
    try:
        for buffer in package.buffers:
            if buffer.data is not None and len(buffer.data) != buffer.size_bytes:
                raise ValueError(f"Buffer {buffer.name} has {len(buffer.data)} bytes, expected {buffer.size_bytes}")
            tensor = device.alloc(buffer.shape, buffer.dtype, buffer.layout, buffer.usage)
            if buffer.data is not None:
                device.write(tensor, buffer.data)
            tensors.append(tensor)

        descriptor = package.serialize_descriptor([tensor.buffer_id for tensor in tensors])
        descriptor_tensor = device.alloc((len(descriptor),), "int8", usage="descriptor")
        device.write(descriptor_tensor, descriptor)
    except Exception:
        if descriptor_tensor is not None:
            try:
                device.free(descriptor_tensor)
            except Exception:
                pass
        for tensor in reversed(tensors):
            try:
                device.free(tensor)
            except Exception:
                pass
        raise

    weights = [
        tensor
        for buffer, tensor in zip(package.buffers, tensors)
        if buffer.usage == BufferUsage.WEIGHT
    ]
    return Model(
        device,
        descriptor_tensor,
        tensors[package.input_index],
        tensors[package.output_index],
        weights=weights,
        tensors=tensors,
    )


def build_mlp_package(
    input_features: int,
    output_features: int,
    weight_values,
    bias_values=None,
    name: str = "mlp",
) -> ModelPackage:
    buffers = [
        HostBuffer(f"{name}.input", (1, input_features), DType.INT8, usage=BufferUsage.ACTIVATION),
        HostBuffer.from_values(
            f"{name}.weight",
            (input_features, output_features),
            DType.INT8,
            weight_values,
            usage=BufferUsage.WEIGHT,
        ),
        HostBuffer(f"{name}.output", (1, output_features), DType.INT32, usage=BufferUsage.OUTPUT),
    ]
    if bias_values is not None:
        buffers.append(
            HostBuffer.from_values(
                f"{name}.bias",
                (output_features,),
                DType.INT32,
                bias_values,
                usage=BufferUsage.WEIGHT,
            )
        )
        bias_index = 3
    else:
        bias_index = 0

    layer = Layer(
        op_type=OpType.GEMM,
        input0=0,
        output=2,
        weight=1,
        bias=bias_index,
        n=1,
        c=input_features,
        out_c=output_features,
    )
    return ModelPackage(buffers=buffers, layers=[layer], input_index=0, output_index=2)


def build_tiny_conv_relu_package(
    input_shape: tuple[int, int, int, int],
    weight_values,
    bias_values=None,
    kernel_shape: tuple[int, int] = (2, 2),
    out_channels: int = 1,
    name: str = "tiny_cnn",
) -> ModelPackage:
    n, c, h, w = input_shape
    kernel_h, kernel_w = kernel_shape
    out_h = h - kernel_h + 1
    out_w = w - kernel_w + 1
    if n <= 0 or c <= 0 or h <= 0 or w <= 0 or out_channels <= 0 or out_h <= 0 or out_w <= 0:
        raise ValueError("Invalid tiny CNN dimensions")

    graph = GraphBuilder(name)
    input_idx = graph.add_buffer("input", input_shape, DType.INT8, usage=BufferUsage.ACTIVATION, layout=Layout.NCHW)
    weight_idx = graph.add_buffer(
        "conv.weight",
        (out_channels, c, kernel_h, kernel_w),
        DType.INT8,
        usage=BufferUsage.WEIGHT,
        data=weight_values,
        layout=Layout.NCHW,
    )
    bias_idx = 0
    if bias_values is not None:
        bias_idx = graph.add_buffer("conv.bias", (out_channels,), DType.INT32, usage=BufferUsage.WEIGHT, data=bias_values)
    conv_out_idx = graph.add_buffer("conv.out", (n, out_channels, out_h, out_w), DType.INT32, usage=BufferUsage.ACTIVATION, layout=Layout.NCHW)
    output_idx = graph.add_buffer("output", (n, out_channels, out_h, out_w), DType.INT32, usage=BufferUsage.OUTPUT, layout=Layout.NCHW)
    graph.conv2d(
        input_idx,
        weight_idx,
        conv_out_idx,
        n=n,
        c=c,
        h=h,
        w=w,
        out_c=out_channels,
        out_h=out_h,
        out_w=out_w,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        bias_idx=bias_idx,
    )
    graph.relu(conv_out_idx, output_idx)
    return graph.package(input_idx, output_idx)


def _flatten(values) -> list:
    if isinstance(values, (bytes, bytearray, memoryview)):
        return list(values)
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


def _num_elements(shape: Iterable[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total
