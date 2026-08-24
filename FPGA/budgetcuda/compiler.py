from __future__ import annotations

import struct
from dataclasses import dataclass, field, replace
from typing import Iterable

from .dtypes import BufferUsage, DType, Layout, OpType, dtype_size, normalize_dtype, normalize_layout
from .model import Model
from .packing import pack_transformer_activations_int8, pack_transformer_weights


MODEL_MAGIC = 0x444D4342
MODEL_VERSION = 1
CONV_FLAG_PRESLICED_WEIGHTS = 0x00000001
CONV_WEIGHT_TILE_DIM = 16
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
        shape = tuple(shape)
        if layout == Layout.TRANSFORMER_ACTIVATION_TILED_INT8:
            if dtype != DType.INT8 or len(shape) != 2:
                raise ValueError("Tiled Transformer activations require a rank-2 INT8 tensor")
            raw = pack_transformer_activations_int8(values, shape[0], shape[1])
        elif layout in (Layout.TRANSFORMER_WEIGHT_TILED_INT8, Layout.TRANSFORMER_WEIGHT_TILED_INT4):
            if dtype != DType.INT8 or len(shape) != 2:
                raise ValueError("Tiled Transformer weights require a rank-2 quantized tensor")
            raw = pack_transformer_weights(
                values, shape[0], shape[1],
                int4=layout == Layout.TRANSFORMER_WEIGHT_TILED_INT4,
            )
        else:
            itemsize = dtype_size(dtype)
            flat = _flatten(values)
            raw = b"".join(int(value).to_bytes(itemsize, "little", signed=True) for value in flat)
        return cls(name, shape, dtype, layout, usage, raw)

    @property
    def size_bytes(self) -> int:
        if len(self.shape) == 2:
            rows, columns = self.shape
            if self.layout == Layout.TRANSFORMER_ACTIVATION_TILED_INT8:
                return _ceil_div(rows, 8) * columns * 8
            if self.layout == Layout.TRANSFORMER_WEIGHT_TILED_INT8:
                return _ceil_div(columns, 8) * rows * 8
            if self.layout == Layout.TRANSFORMER_WEIGHT_TILED_INT4:
                return _ceil_div(columns, 8) * rows * 4
            if self.layout == Layout.TRANSFORMER_DYNAMIC_B_TILED_INT8:
                return _ceil_div(columns, 8) * rows * 8
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


class ResidentBufferPool:
    """Own named device tensors that can be borrowed by multiple graph uploads."""

    def __init__(self, device):
        self.device = device
        self.tensors = {}
        self.specifications = {}
        self._closed = False

    @property
    def bytes(self) -> int:
        return sum(tensor.size_bytes for tensor in self.tensors.values())

    def upload(self, buffers: Iterable[HostBuffer]) -> None:
        if self._closed:
            raise RuntimeError("Resident buffer pool is closed")
        for buffer in buffers:
            if buffer.data is None:
                raise ValueError(f"Resident buffer {buffer.name} has no initial data")
            specification = (buffer.shape, buffer.dtype, buffer.layout, buffer.usage, buffer.size_bytes)
            existing = self.specifications.get(buffer.name)
            if existing is not None:
                if existing != specification:
                    raise ValueError(f"Incompatible resident buffer reuse: {buffer.name}")
                continue
            if len(buffer.data) != buffer.size_bytes:
                raise ValueError(f"Buffer {buffer.name} has {len(buffer.data)} bytes, expected {buffer.size_bytes}")
            tensor = self.device.alloc(buffer.shape, buffer.dtype, buffer.layout, buffer.usage)
            try:
                self.device.write(tensor, buffer.data)
            except Exception:
                self.device.free(tensor)
                raise
            self.tensors[buffer.name] = tensor
            self.specifications[buffer.name] = specification

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error = None
        for tensor in reversed(list(self.tensors.values())):
            try:
                self.device.free(tensor)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        self.tensors.clear()
        self.specifications.clear()
        if first_error is not None:
            raise first_error


def compose_model_packages(name: str, packages: Iterable[ModelPackage]) -> ModelPackage:
    """Fuse sequential packages, aliasing each output to the next input."""
    packages = list(packages)
    if not packages:
        raise ValueError("At least one package is required")
    buffers = []
    layers = []
    previous_output = None
    first_input = None
    for package_number, package in enumerate(packages):
        mapping = {}
        if previous_output is not None:
            expected = package.buffers[package.input_index]
            actual = buffers[previous_output]
            if (expected.dtype != actual.dtype or expected.layout != actual.layout or
                    expected.size_bytes > actual.size_bytes):
                raise ValueError(
                    f"Package {package_number} input is incompatible with the previous output")
            mapping[package.input_index] = previous_output
        for index, buffer in enumerate(package.buffers):
            if index in mapping:
                continue
            mapping[index] = len(buffers)
            buffers.append(buffer)
        if first_input is None:
            first_input = mapping[package.input_index]

        def required(index: int) -> int:
            return mapping[index]

        def optional(index: int) -> int:
            return 0 if index == 0 else mapping[index]

        for layer in package.layers:
            layers.append(replace(
                layer,
                input0=required(layer.input0),
                input1=optional(layer.input1),
                input2=optional(layer.input2),
                output=required(layer.output),
                weight=optional(layer.weight),
                bias=optional(layer.bias),
                scale=optional(layer.scale),
            ))
        previous_output = mapping[package.output_index]
    return ModelPackage(
        buffers, layers, first_input, previous_output,
        flags=packages[-1].flags,
    )


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

    def add_transformer_activation(self, name: str, rows: int, features: int, *, data=None, usage: BufferUsage = BufferUsage.ACTIVATION) -> int:
        if rows <= 0 or features <= 0:
            raise ValueError("Transformer activation dimensions must be positive")
        return self.add_buffer(
            name, (rows, features), DType.INT8, usage, data,
            layout=Layout.TRANSFORMER_ACTIVATION_TILED_INT8,
        )

    def add_transformer_weight(self, name: str, reduction: int, outputs: int, values, *, int4: bool = False) -> int:
        if reduction <= 0 or outputs <= 0:
            raise ValueError("Transformer weight dimensions must be positive")
        layout = Layout.TRANSFORMER_WEIGHT_TILED_INT4 if int4 else Layout.TRANSFORMER_WEIGHT_TILED_INT8
        return self.add_buffer(name, (reduction, outputs), DType.INT8, BufferUsage.WEIGHT, values, layout=layout)

    def add_transformer_dynamic_b(self, name: str, reduction: int, outputs: int) -> int:
        if reduction <= 0 or outputs <= 0:
            raise ValueError("Dynamic Transformer B dimensions must be positive")
        return self.add_buffer(
            name, (reduction, outputs), DType.INT8,
            layout=Layout.TRANSFORMER_DYNAMIC_B_TILED_INT8,
        )

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
        preslice_weights: bool = True,
    ) -> None:
        flags = 0
        if preslice_weights:
            weight = self.buffers[weight_idx]
            reduction = c * kernel_h * kernel_w
            tiled_shape = (
                _ceil_div(out_c, CONV_WEIGHT_TILE_DIM),
                _ceil_div(reduction, CONV_WEIGHT_TILE_DIM),
                CONV_WEIGHT_TILE_DIM,
                CONV_WEIGHT_TILE_DIM,
            )
            if weight.layout == Layout.CONV_WEIGHT_TILED_INT8:
                if weight.shape != tiled_shape:
                    raise ValueError(f"Presliced Conv2D weight {weight.name} has incompatible shape {weight.shape}")
            else:
                if weight.dtype != DType.INT8 or weight.data is None:
                    raise ValueError("Host Conv2D preslicing requires constant INT8 weights")
                self.buffers[weight_idx] = HostBuffer(
                    weight.name,
                    tiled_shape,
                    weight.dtype,
                    Layout.CONV_WEIGHT_TILED_INT8,
                    weight.usage,
                    preslice_conv2d_weights_int8(
                        weight.data,
                        out_channels=out_c,
                        input_channels=c,
                        kernel_h=kernel_h,
                        kernel_w=kernel_w,
                    ),
                )
            flags |= CONV_FLAG_PRESLICED_WEIGHTS
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
                flags=flags,
            )
        )

    def gemm(self, input_idx: int, weight_idx: int, output_idx: int, *, m: int, n: int, k: int, bias_idx: int = 0) -> None:
        self.layers.append(Layer(OpType.GEMM, input0=input_idx, weight=weight_idx, output=output_idx,
                                 bias=bias_idx, n=m, out_c=n, c=k))

    def transformer_gemm(self, input_idx: int, weight_idx: int, output_idx: int, *, m: int, n: int, k: int, bias_idx: int = 0) -> None:
        activation = self.buffers[input_idx]
        weight = self.buffers[weight_idx]
        output = self.buffers[output_idx]
        if activation.layout != Layout.TRANSFORMER_ACTIVATION_TILED_INT8 or activation.shape != (m, k):
            raise ValueError("Transformer GEMM activation must use the matching tiled MxK layout")
        if weight.layout not in (Layout.TRANSFORMER_WEIGHT_TILED_INT8, Layout.TRANSFORMER_WEIGHT_TILED_INT4,
                                 Layout.TRANSFORMER_DYNAMIC_B_TILED_INT8) or weight.shape != (k, n):
            raise ValueError("Transformer GEMM weight must use the matching tiled KxN layout")
        if output.dtype != DType.INT32 or output.layout != Layout.ROW_MAJOR or output.shape != (m, n):
            raise ValueError("Transformer GEMM output must be row-major INT32 MxN")
        if bias_idx:
            bias = self.buffers[bias_idx]
            if bias.dtype != DType.INT32 or bias.layout != Layout.ROW_MAJOR or bias.shape != (n,):
                raise ValueError("Transformer GEMM bias must be row-major INT32 with N elements")
        self.gemm(input_idx, weight_idx, output_idx, m=m, n=n, k=k, bias_idx=bias_idx)

    def transformer_gemm_requant(
        self, input_idx: int, weight_idx: int, output_idx: int, scale_idx: int,
        *, m: int, n: int, k: int, bias_idx: int = 0,
        dynamic_b_transpose: bool = False,
    ) -> None:
        """Fuse tiled GEMM, optional INT32 bias, requantize, and A packing."""
        activation = self.buffers[input_idx]
        weight = self.buffers[weight_idx]
        output = self.buffers[output_idx]
        scale = self.buffers[scale_idx]
        if activation.layout != Layout.TRANSFORMER_ACTIVATION_TILED_INT8 or activation.shape != (m, k):
            raise ValueError("Fused Transformer GEMM activation must use matching tiled MxK INT8")
        if weight.layout not in (
            Layout.TRANSFORMER_WEIGHT_TILED_INT8,
            Layout.TRANSFORMER_WEIGHT_TILED_INT4,
            Layout.TRANSFORMER_DYNAMIC_B_TILED_INT8,
        ) or weight.shape != (k, n):
            raise ValueError("Fused Transformer GEMM weight must use matching tiled KxN layout")
        valid_output = (
            output.dtype == DType.INT8 and (
                (output.layout in (Layout.TRANSFORMER_ACTIVATION_TILED_INT8, Layout.ROW_MAJOR)
                 and output.shape == (m, n)) or
                (output.layout == Layout.TRANSFORMER_DYNAMIC_B_TILED_INT8 and
                 output.shape == ((n, m) if dynamic_b_transpose else (m, n)))
            )
        )
        if not valid_output:
            raise ValueError("Fused Transformer GEMM output layout/shape does not match MxN")
        if scale.dtype != DType.INT32 or scale.layout != Layout.ROW_MAJOR or scale.shape != (4,):
            raise ValueError("Fused Transformer GEMM requant parameters must contain four INT32 values")
        if bias_idx:
            bias = self.buffers[bias_idx]
            if bias.dtype != DType.INT32 or bias.layout != Layout.ROW_MAJOR or bias.shape != (n,):
                raise ValueError("Fused Transformer GEMM bias must be row-major INT32 with N elements")
        self.layers.append(Layer(
            OpType.GEMM, input0=input_idx, weight=weight_idx, output=output_idx,
            bias=bias_idx, scale=scale_idx, n=m, out_c=n, c=k,
            flags=1 if output.layout == Layout.TRANSFORMER_DYNAMIC_B_TILED_INT8 and dynamic_b_transpose else 0,
        ))

    def activation_b_pack(self, input_idx: int, output_idx: int, *, reduction: int, outputs: int, transpose: bool = False) -> None:
        source = self.buffers[input_idx]
        output = self.buffers[output_idx]
        expected_source_shape = (outputs, reduction) if transpose else (reduction, outputs)
        if source.dtype != DType.INT8 or source.layout != Layout.TRANSFORMER_ACTIVATION_TILED_INT8 or source.shape != expected_source_shape:
            raise ValueError("Activation B-pack source must be the matching tiled INT8 activation")
        if output.dtype != DType.INT8 or output.layout != Layout.TRANSFORMER_DYNAMIC_B_TILED_INT8 or output.shape != (reduction, outputs):
            raise ValueError("Activation B-pack output must be the matching dynamic B tensor")
        self.layers.append(Layer(OpType.ACTIVATION_B_PACK, input0=input_idx, output=output_idx,
                                 n=reduction, c=outputs, flags=1 if transpose else 0))

    def activation_a_pack(self, input_idx: int, output_idx: int, *, rows: int, features: int) -> None:
        source = self.buffers[input_idx]
        output = self.buffers[output_idx]
        if source.dtype != DType.INT8 or source.layout != Layout.ROW_MAJOR or source.shape != (rows, features):
            raise ValueError("Activation A-pack source must be matching row-major INT8")
        if output.dtype != DType.INT8 or output.layout != Layout.TRANSFORMER_ACTIVATION_TILED_INT8 or output.shape != (rows, features):
            raise ValueError("Activation A-pack output must be matching tiled INT8")
        self.layers.append(Layer(OpType.ACTIVATION_B_PACK, input0=input_idx, output=output_idx,
                                 n=features, c=rows, flags=3))

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

    def requantize_pack(self, input_idx: int, output_idx: int, scale_idx: int, *, rows: int, columns: int) -> None:
        source = self.buffers[input_idx]
        output = self.buffers[output_idx]
        if source.dtype != DType.INT32 or source.layout != Layout.ROW_MAJOR or source.shape != (rows, columns):
            raise ValueError("Requantize-pack input must be row-major INT32 with the requested shape")
        if output.dtype != DType.INT8 or output.layout != Layout.TRANSFORMER_ACTIVATION_TILED_INT8 or output.shape != (rows, columns):
            raise ValueError("Requantize-pack output must be a matching tiled INT8 activation")
        self.layers.append(Layer(OpType.REQUANTIZE, input0=input_idx, output=output_idx, scale=scale_idx, n=rows, c=columns))

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

    def avgpool2d(self, input_idx: int, output_idx: int, **kwargs) -> None:
        self.maxpool2d(input_idx, output_idx, **kwargs)
        self.layers[-1] = replace(self.layers[-1], flags=self.layers[-1].flags | 1)

    def global_avgpool2d(self, input_idx: int, output_idx: int, *, n: int, c: int, h: int, w: int) -> None:
        self.avgpool2d(input_idx, output_idx, n=n, c=c, h=h, w=w, out_h=1, out_w=1, kernel_h=h, kernel_w=w, stride_h=1, stride_w=1)

    def upsample(self, input_idx: int, output_idx: int, *, n: int, c: int, h: int, w: int, out_h: int, out_w: int) -> None:
        self.layers.append(Layer(OpType.UPSAMPLE, input0=input_idx, output=output_idx, n=n, c=c, h=h, w=w, out_h=out_h, out_w=out_w))

    def concat_channels(self, input0: int, input1: int, output_idx: int, *, n: int, h: int, w: int, out_c: int, input2: int = 0) -> None:
        self.layers.append(Layer(OpType.CONCAT, input0=input0, input1=input1, input2=input2, output=output_idx, n=n, h=h, w=w, out_c=out_c))

    def reshape(self, input_idx: int, output_idx: int) -> None:
        self.layers.append(Layer(OpType.RESHAPE, input0=input_idx, output=output_idx))

    def layer_norm(self, input_idx: int, gamma_idx: int, beta_idx: int, output_idx: int, *, rows: int, features: int, epsilon_term: int = 0) -> None:
        if rows <= 0 or features <= 0 or epsilon_term < 0 or epsilon_term > 0xFFFFFFFF:
            raise ValueError("Invalid LayerNorm dimensions or epsilon term")
        self.layers.append(Layer(OpType.LAYERNORM, input0=input_idx, output=output_idx, scale=gamma_idx, bias=beta_idx, n=rows, c=features, flags=epsilon_term))

    def gelu(self, input_idx: int, output_idx: int) -> None:
        self.layers.append(Layer(OpType.GELU, input0=input_idx, output=output_idx))

    def softmax(self, input_idx: int, output_idx: int, *, rows: int, length: int) -> None:
        if rows <= 0 or length <= 0:
            raise ValueError("Invalid Softmax dimensions")
        source, output = self.buffers[input_idx], self.buffers[output_idx]
        if source.dtype != DType.INT8 or source.layout != Layout.ROW_MAJOR or source.shape != (rows, length):
            raise ValueError("Softmax input must be matching row-major INT8")
        if output.dtype != DType.INT8 or output.layout not in (
            Layout.ROW_MAJOR, Layout.TRANSFORMER_ACTIVATION_TILED_INT8,
        ) or output.shape != (rows, length):
            raise ValueError("Softmax output must be matching row-major or tiled INT8")
        self.layers.append(Layer(OpType.SOFTMAX, input0=input_idx, output=output_idx, n=rows, c=length))

    def residual_add(self, lhs_idx: int, rhs_idx: int, output_idx: int, *, lhs_shift: int = 0, rhs_shift: int = 0) -> None:
        if lhs_shift not in range(8) or rhs_shift not in range(8):
            raise ValueError("Residual shifts must be in 0..7")
        self.layers.append(Layer(OpType.RESIDUAL_ADD, input0=lhs_idx, input1=rhs_idx, output=output_idx, flags=lhs_shift | (rhs_shift << 4)))

    def token_concat(self, inputs: list[int], output_idx: int, *, rows: int, widths: list[int]) -> None:
        if not 1 <= len(inputs) <= 3 or len(inputs) != len(widths) or rows <= 0 or any(width <= 0 for width in widths):
            raise ValueError("Token concat requires one to three positive-width inputs")
        for index, width in zip(inputs, widths):
            buffer = self.buffers[index]
            if buffer.dtype != DType.INT8 or buffer.layout != Layout.ROW_MAJOR or buffer.shape != (rows, width):
                raise ValueError("Token concat inputs must be matching row-major INT8 tensors")
        output = self.buffers[output_idx]
        if output.dtype != DType.INT8 or output.layout not in (
            Layout.ROW_MAJOR, Layout.TRANSFORMER_ACTIVATION_TILED_INT8,
        ) or output.shape != (rows, sum(widths)):
            raise ValueError("Token concat output must be matching row-major or tiled INT8")
        padded_inputs = inputs + [0] * (3 - len(inputs))
        padded_widths = widths + [0] * (3 - len(widths))
        self.layers.append(Layer(OpType.TOKEN_CONCAT, input0=padded_inputs[0], input1=padded_inputs[1],
                                 input2=padded_inputs[2], output=output_idx, n=rows,
                                 c=padded_widths[0], h=padded_widths[1], w=padded_widths[2], out_c=sum(widths)))

    def token_embedding(
        self, patches_idx: int, position_idx: int, scale_idx: int,
        output_idx: int, *, patch_rows: int, features: int,
    ) -> None:
        patches = self.buffers[patches_idx]
        position = self.buffers[position_idx]
        scale = self.buffers[scale_idx]
        output = self.buffers[output_idx]
        if (patches.dtype != DType.INT8 or patches.layout != Layout.ROW_MAJOR or
                patches.shape != (patch_rows, features)):
            raise ValueError("Token embedding patches must be row-major INT8")
        if (position.dtype != DType.INT8 or position.layout != Layout.ROW_MAJOR or
                position.shape != (patch_rows + 1, features)):
            raise ValueError("Token embedding position tensor has the wrong shape")
        if scale.dtype != DType.INT8 or scale.shape != (256,):
            raise ValueError("Token embedding scale must be a 256-entry INT8 LUT")
        if (output.dtype != DType.INT8 or output.layout != Layout.ROW_MAJOR or
                output.shape != (patch_rows + 1, features)):
            raise ValueError("Token embedding output has the wrong shape")
        self.layers.append(Layer(
            OpType.TOKEN_EMBEDDING, input0=patches_idx, output=output_idx,
            weight=position_idx, scale=scale_idx, n=patch_rows, c=features,
        ))

    def package(self, input_idx: int, output_idx: int) -> ModelPackage:
        return ModelPackage(self.buffers, self.layers, input_idx, output_idx)


def upload_model(device, package: ModelPackage, external_tensors=None) -> Model:
    external_tensors = {} if external_tensors is None else external_tensors
    tensors = []
    owned_tensors = []
    descriptor_tensor = None
    try:
        for buffer in package.buffers:
            if buffer.data is not None and len(buffer.data) != buffer.size_bytes:
                raise ValueError(f"Buffer {buffer.name} has {len(buffer.data)} bytes, expected {buffer.size_bytes}")
            tensor = external_tensors.get(buffer.name)
            if tensor is not None:
                if (tuple(tensor.shape) != buffer.shape or tensor.dtype != buffer.dtype or
                        tensor.layout != buffer.layout or tensor.size_bytes != buffer.size_bytes):
                    raise ValueError(f"External tensor does not match buffer {buffer.name}")
            else:
                tensor = device.alloc(buffer.shape, buffer.dtype, buffer.layout, buffer.usage)
                owned_tensors.append(tensor)
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
        for tensor in reversed(owned_tensors):
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
        tensors=owned_tensors,
        package=package,
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
    preslice_weights: bool = True,
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
        preslice_weights=preslice_weights,
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


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + divisor - 1) // divisor


def preslice_conv2d_weights_int8(
    values,
    *,
    out_channels: int,
    input_channels: int,
    kernel_h: int,
    kernel_w: int,
    tile_dim: int = CONV_WEIGHT_TILE_DIM,
) -> bytes:
    """Pack OIHW INT8 weights into tile-major B[K lane][N lane] order."""
    if min(out_channels, input_channels, kernel_h, kernel_w, tile_dim) <= 0:
        raise ValueError("Conv2D preslice dimensions must be positive")
    raw = bytes(values) if isinstance(values, (bytes, bytearray, memoryview)) else bytes(
        int(value) & 0xFF for value in _flatten(values)
    )
    reduction = input_channels * kernel_h * kernel_w
    expected = out_channels * reduction
    if len(raw) != expected:
        raise ValueError(f"Conv2D weight has {len(raw)} bytes, expected {expected}")

    packed = bytearray(_ceil_div(out_channels, tile_dim) * _ceil_div(reduction, tile_dim) * tile_dim * tile_dim)
    dst = 0
    for n_tile in range(_ceil_div(out_channels, tile_dim)):
        for k_tile in range(_ceil_div(reduction, tile_dim)):
            for k_lane in range(tile_dim):
                k = k_tile * tile_dim + k_lane
                for n_lane in range(tile_dim):
                    oc = n_tile * tile_dim + n_lane
                    if oc < out_channels and k < reduction:
                        packed[dst] = raw[oc * reduction + k]
                    dst += 1
    return bytes(packed)
