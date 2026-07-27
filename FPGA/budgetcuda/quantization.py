from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from .dtypes import DType, normalize_dtype


@dataclass(frozen=True)
class QuantizationParams:
    """Scalar signed-INT8 affine quantization parameters for one tensor."""

    scale: float
    zero_point: int = 0

    def __post_init__(self):
        if not math.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise ValueError("Quantization scale must be positive and finite")
        if int(self.zero_point) < -128 or int(self.zero_point) > 127:
            raise ValueError("Signed INT8 zero point must be in -128..127")

    def quantize(self, values):
        return quantize_int8(values, self.scale, self.zero_point)

    def dequantize(self, values):
        return dequantize_int8(values, self.scale, self.zero_point)


@dataclass(frozen=True)
class Int4QuantizationParams:
    """Scalar signed-INT4 affine quantization parameters."""

    scale: float
    zero_point: int = 0

    def __post_init__(self):
        if not math.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise ValueError("Quantization scale must be positive and finite")
        if int(self.zero_point) < -8 or int(self.zero_point) > 7:
            raise ValueError("Signed INT4 zero point must be in -8..7")

    def quantize(self, values):
        return quantize_int4(values, self.scale, self.zero_point)

    def dequantize(self, values):
        return dequantize_int4(values, self.scale, self.zero_point)


@dataclass(frozen=True)
class ModelQDQParams:
    """Input and output Q/DQ metadata extracted from a deployable ONNX model."""

    input: QuantizationParams | Int4QuantizationParams
    output: QuantizationParams | Int4QuantizationParams
    input_name: str
    output_name: str


def onnx_qdq_params(source, *, target_dtype: str | int | DType = DType.INT4) -> ModelQDQParams:
    """Extract scalar input/output Q/DQ parameters from an ONNX model.

    ``source`` may be a path or an ``onnx.ModelProto``. The model must expose
    one runtime input feeding ``DequantizeLinear`` and one graph output produced
    by ``DequantizeLinear``. Per-channel parameters are rejected because the
    deployed BudgetCUDA datapath uses per-tensor quantization.
    """

    try:
        import onnx
        from onnx import numpy_helper
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ONNX Q/DQ inspection requires the optional dependency 'onnx'. "
            "Install it with: python -m pip install onnx"
        ) from exc

    target_dtype = normalize_dtype(target_dtype)
    if target_dtype not in (DType.INT4, DType.INT8):
        raise ValueError("ONNX Q/DQ parameters are only supported for INT4 or INT8")
    model = onnx.load(str(source)) if isinstance(source, (str, Path)) else source
    graph = model.graph
    initializers = {value.name: numpy_helper.to_array(value) for value in graph.initializer}
    runtime_inputs = [value for value in graph.input if value.name not in initializers]
    if len(runtime_inputs) != 1 or len(graph.output) != 1:
        raise ValueError("BudgetCUDA Q/DQ inspection requires one runtime input and one output")

    input_name = runtime_inputs[0].name
    input_dq = next(
        (
            node
            for node in graph.node
            if node.op_type == "DequantizeLinear" and node.input and node.input[0] == input_name
        ),
        None,
    )
    if input_dq is None:
        raise ValueError(f"ONNX input {input_name!r} must feed DequantizeLinear")

    producers = {name: node for node in graph.node for name in node.output}
    output_name = graph.output[0].name
    output_dq = producers.get(output_name)
    if output_dq is None or output_dq.op_type != "DequantizeLinear":
        raise ValueError(f"ONNX output {output_name!r} must be produced by DequantizeLinear")

    return ModelQDQParams(
        input=_params_from_dq(input_dq, initializers, f"input {input_name!r}", target_dtype),
        output=_params_from_dq(output_dq, initializers, f"output {output_name!r}", target_dtype),
        input_name=input_name,
        output_name=output_name,
    )


def quantize_int8(values, scale: float, zero_point: int = 0):
    """Quantize values to signed INT8 while preserving NumPy array shape."""

    params = QuantizationParams(scale, zero_point)
    try:
        import numpy as np

        values_array = np.asarray(values, dtype=np.float32)
        quantized = np.rint(values_array / params.scale + params.zero_point)
        return np.clip(quantized, -128, 127).astype(np.int8)
    except ModuleNotFoundError:
        return [
            max(-128, min(127, int(round(float(value) / params.scale + params.zero_point))))
            for value in _flatten(values)
        ]


def dequantize_int8(values, scale: float, zero_point: int = 0):
    """Dequantize signed INT8 values while preserving NumPy array shape."""

    params = QuantizationParams(scale, zero_point)
    try:
        import numpy as np

        return (np.asarray(values, dtype=np.float32) - params.zero_point) * params.scale
    except ModuleNotFoundError:
        return [(int(value) - params.zero_point) * params.scale for value in _flatten(values)]


def quantize_int4(values, scale: float, zero_point: int = 0):
    """Truncate values to signed INT4 represented in an INT8 host container."""

    params = Int4QuantizationParams(scale, zero_point)
    try:
        import numpy as np

        values_array = np.asarray(values, dtype=np.float32)
        quantized = np.trunc(values_array / params.scale + params.zero_point)
        return np.clip(quantized, -8, 7).astype(np.int8)
    except ModuleNotFoundError:
        return [
            max(-8, min(7, int(float(value) / params.scale + params.zero_point)))
            for value in _flatten(values)
        ]


def dequantize_int4(values, scale: float, zero_point: int = 0):
    """Dequantize signed INT4 values while preserving NumPy array shape."""

    params = Int4QuantizationParams(scale, zero_point)
    try:
        import numpy as np

        return (np.asarray(values, dtype=np.float32) - params.zero_point) * params.scale
    except ModuleNotFoundError:
        return [(int(value) - params.zero_point) * params.scale for value in _flatten(values)]


def quantize_symmetric_int4(values, scale: float | None = None):
    flat = _flatten(values)
    if scale is None:
        max_abs = max((abs(float(value)) for value in flat), default=0.0)
        scale = 1.0 if max_abs == 0.0 else max_abs / 7.0
    q = [max(-8, min(7, int(round(float(value) / float(scale))))) for value in flat]
    return _maybe_numpy_array(q, "int8"), float(scale)


def dequantize_symmetric_int4(values, scale: float):
    out = [int(value) * float(scale) for value in _flatten(values)]
    return _maybe_numpy_array(out, "float32")


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


def _params_from_dq(
    node,
    initializers,
    label: str,
    target_dtype: DType,
) -> QuantizationParams | Int4QuantizationParams:
    if len(node.input) < 3:
        raise ValueError(f"ONNX DequantizeLinear for {label} requires scale and zero point")
    try:
        scale_array = initializers[node.input[1]]
        zero_point_array = initializers[node.input[2]]
    except KeyError as exc:
        raise ValueError(f"ONNX Q/DQ parameters for {label} must be constant initializers") from exc
    if scale_array.size != 1 or zero_point_array.size != 1:
        raise ValueError(f"ONNX Q/DQ parameters for {label} must be scalar per-tensor values")
    params_type = Int4QuantizationParams if target_dtype == DType.INT4 else QuantizationParams
    return params_type(
        float(scale_array.reshape(-1)[0]),
        int(zero_point_array.reshape(-1)[0]),
    )
