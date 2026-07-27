"""ONNX-to-BudgetCUDA lowering for the deployable signed inference subset.

The importer intentionally accepts already-quantized, static-shape graphs.  It
does not silently quantize float graphs: quantization belongs in the training
or export pipeline, where calibration information is available.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from .compiler import GraphBuilder, HostBuffer, Layer, ModelPackage
from .dtypes import BufferUsage, DType, Layout, OpType, normalize_dtype


class OnnxImportError(ValueError):
    """Raised when an ONNX graph cannot be represented by the BudgetCUDA IR."""


def import_onnx_model(
    source,
    *,
    name: str | None = None,
    input_shapes: Mapping[str, tuple[int, ...]] | None = None,
    target_dtype: str | DType = DType.INT4,
) -> ModelPackage:
    """Load a static signed Q/DQ ONNX graph into a ``ModelPackage``.

    ``source`` may be a path or an ``onnx.ModelProto``.  The optional
    ``input_shapes`` mapping supplies concrete dimensions for graph inputs that
    were exported with symbolic dimensions.

    Packed INT4 is the production default. ONNX carries those values in signed
    INT8 containers, and the importer verifies every constant is in ``[-8, 7]``
    before packing it. Pass ``target_dtype="int8"`` only for legacy images.

    Supported nodes are ``QLinearConv``, INT8/INT32 ``Conv``, ``Relu``, a
    power-of-two ``LeakyRelu``, ``MaxPool``, ``Flatten``, shape-only
    ``Reshape``, ``MatMul``, and a restricted ``Gemm``.  QLinearConv requires
    signed INT8 activations and weights with zero point zero; it is lowered to
    INT32 convolution plus the existing fixed-point requantizer.
    """
    try:
        target_dtype = normalize_dtype(target_dtype)
    except (TypeError, ValueError) as exc:
        raise OnnxImportError("ONNX deployment target must be INT4 or INT8") from exc
    if target_dtype not in (DType.INT4, DType.INT8):
        raise OnnxImportError("ONNX deployment target must be INT4 or INT8")
    onnx, numpy_helper = _require_onnx()
    if isinstance(source, (str, Path)):
        model = onnx.load(str(source))
    else:
        model = source
    try:
        onnx.checker.check_model(model)
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        raise OnnxImportError(f"Invalid or non-inferable ONNX model: {exc}") from exc

    graph_proto = model.graph
    if any(node.op_type == "DequantizeLinear" for node in graph_proto.node):
        package = _import_signed_qdq_model(model, package_name=name or graph_proto.name or "onnx_model", input_shapes=input_shapes or {})
        return _convert_qdq_package_to_int4(package) if target_dtype == DType.INT4 else package
    if target_dtype == DType.INT4:
        raise OnnxImportError("INT4 deployment requires a signed Q/DQ QAT model with values already constrained to -8..7")
    package_name = name or graph_proto.name or "onnx_model"
    shapes = _collect_shapes(graph_proto, input_shapes or {})
    initializers = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph_proto.initializer}
    initializer_names = set(initializers)
    graph_inputs = [value for value in graph_proto.input if value.name not in initializer_names]
    if len(graph_inputs) != 1:
        raise OnnxImportError("BudgetCUDA currently imports models with exactly one runtime input")
    graph_outputs = [value.name for value in graph_proto.output]
    if len(graph_outputs) != 1:
        raise OnnxImportError("BudgetCUDA currently imports models with exactly one model output")

    builder = GraphBuilder(package_name)
    buffers: dict[str, int] = {}
    input_name = graph_inputs[0].name
    input_shape = _shape_for(shapes, input_name)
    input_dtype = _onnx_dtype(graph_inputs[0].type.tensor_type.elem_type, onnx)
    if input_dtype != DType.INT8:
        raise OnnxImportError("Model input must be signed INT8; export a quantized INT8 ONNX model")
    buffers[input_name] = builder.add_buffer(input_name, input_shape, input_dtype, BufferUsage.ACTIVATION, layout=_layout(input_shape))

    def add_initializer(tensor_name: str) -> int:
        if tensor_name in buffers:
            return buffers[tensor_name]
        try:
            array = initializers[tensor_name]
        except KeyError as exc:
            raise OnnxImportError(f"{tensor_name!r} must be a constant initializer") from exc
        dtype = _array_dtype(array, onnx)
        if dtype not in (DType.INT8, DType.INT32):
            raise OnnxImportError(f"Initializer {tensor_name!r} must be INT8 or INT32")
        buffers[tensor_name] = builder.add_buffer(
            tensor_name,
            tuple(int(dim) for dim in array.shape) or (1,),
            dtype,
            BufferUsage.WEIGHT,
            data=array,
            layout=_layout(tuple(int(dim) for dim in array.shape)),
        )
        return buffers[tensor_name]

    def add_activation(tensor_name: str, dtype: DType, *, output: bool = False) -> int:
        if tensor_name in buffers:
            return buffers[tensor_name]
        buffers[tensor_name] = builder.add_buffer(
            tensor_name,
            _shape_for(shapes, tensor_name),
            dtype,
            BufferUsage.OUTPUT if output else BufferUsage.ACTIVATION,
            layout=_layout(_shape_for(shapes, tensor_name)),
        )
        return buffers[tensor_name]

    for node in graph_proto.node:
        op = node.op_type
        if op == "QLinearConv":
            _lower_qlinear_conv(node, builder, buffers, add_initializer, add_activation, initializers, shapes, graph_outputs, onnx)
        elif op == "Conv":
            _lower_conv(node, builder, buffers, add_initializer, add_activation, initializers, shapes, graph_outputs, onnx)
        elif op in ("Relu", "LeakyRelu"):
            _lower_activation(node, builder, buffers, add_activation, shapes, graph_outputs, onnx)
        elif op in ("MaxPool", "AveragePool", "GlobalAveragePool"):
            _lower_maxpool(node, builder, buffers, add_activation, shapes, graph_outputs, onnx)
        elif op in ("Flatten", "Reshape"):
            _lower_reshape(node, builder, buffers, add_activation, shapes, graph_outputs, onnx)
        elif op in ("MatMul", "Gemm"):
            _lower_gemm(node, builder, buffers, add_initializer, add_activation, initializers, shapes, graph_outputs, onnx)
        else:
            raise OnnxImportError(f"Unsupported ONNX node {op!r} ({node.name or 'unnamed'})")

    output_name = graph_outputs[0]
    if output_name not in buffers:
        raise OnnxImportError(f"ONNX graph output {output_name!r} is not produced by a supported node")
    return builder.package(buffers[input_name], buffers[output_name])


def _convert_qdq_package_to_int4(package: ModelPackage) -> ModelPackage:
    """Pack an INT8-container QAT graph whose quantized values are true INT4."""

    converted = []
    for buffer in package.buffers:
        if buffer.dtype != DType.INT8:
            converted.append(buffer)
            continue
        if buffer.data is None:
            converted.append(
                HostBuffer(buffer.name, buffer.shape, DType.INT4, buffer.layout, buffer.usage)
            )
            continue
        values = [byte - 256 if byte & 0x80 else byte for byte in buffer.data]
        out_of_range = next((value for value in values if value < -8 or value > 7), None)
        if out_of_range is not None:
            raise OnnxImportError(
                f"QAT tensor {buffer.name!r} contains {out_of_range}, outside signed INT4; "
                "export with a -8..7 fake-quant range"
            )
        converted.append(
            HostBuffer.from_values(
                buffer.name,
                buffer.shape,
                DType.INT4,
                values,
                usage=buffer.usage,
                layout=buffer.layout,
            )
        )
    return ModelPackage(
        converted,
        list(package.layers),
        package.input_index,
        package.output_index,
        package.activation_arena_a,
        package.activation_arena_b,
        package.workspace_buffer,
        package.flags,
    )


def _import_signed_qdq_model(model, *, package_name: str, input_shapes: Mapping[str, tuple[int, ...]]) -> ModelPackage:
    """Lower the zero-centered, per-tensor Q/DQ dialect used by deployable exports."""
    onnx, numpy_helper = _require_onnx()
    graph = model.graph
    shapes = _collect_shapes(graph, input_shapes)
    initializers = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}
    inputs = [value for value in graph.input if value.name not in initializers]
    if len(inputs) != 1 or inputs[0].type.tensor_type.elem_type != onnx.TensorProto.INT8:
        raise OnnxImportError("Q/DQ lowering requires exactly one signed INT8 runtime input; unsigned PyTorch Q/DQ exports need symmetric-QAT retraining")

    producers = {output: node for node in graph.node for output in node.output}
    consumers: dict[str, list] = {}
    for node in graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)

    def scalar(name: str, label: str, *, positive: bool = False) -> float:
        try:
            value = initializers[name]
        except KeyError as exc:
            raise OnnxImportError(f"Q/DQ {label} must be an initializer") from exc
        if value.size != 1:
            raise OnnxImportError(f"Q/DQ {label} must be per-tensor; per-channel quantization is not implemented in PL")
        result = float(value.reshape(-1)[0])
        if positive and result <= 0:
            raise OnnxImportError(f"Q/DQ {label} must be positive")
        return result

    def qdq_source(value: str) -> tuple[str, float]:
        node = producers.get(value)
        if node is None or node.op_type != "DequantizeLinear":
            raise OnnxImportError("Q/DQ compute inputs must be produced by DequantizeLinear")
        if len(node.input) < 3:
            raise OnnxImportError("Q/DQ DequantizeLinear requires scale and zero point")
        if int(scalar(node.input[2], "zero point")) != 0:
            raise OnnxImportError("Q/DQ zero point must be zero for signed BudgetCUDA execution")
        return node.input[0], scalar(node.input[1], "scale", positive=True)

    def qdq_output(value: str) -> tuple[str, float]:
        nodes = [node for node in consumers.get(value, []) if node.op_type == "QuantizeLinear"]
        if len(nodes) != 1 or len(nodes[0].input) < 3:
            raise OnnxImportError("Each Q/DQ compute result must feed one QuantizeLinear")
        node = nodes[0]
        if int(scalar(node.input[2], "zero point")) != 0:
            raise OnnxImportError("Q/DQ zero point must be zero for signed BudgetCUDA execution")
        quantized = node.output[0]
        clips = [
            consumer
            for consumer in consumers.get(quantized, [])
            if consumer.op_type == "Clip"
        ]
        if clips:
            if len(clips) != 1 or len(clips[0].input) != 3:
                raise OnnxImportError(
                    "Q/DQ INT4 QuantizeLinear must feed one Clip with constant bounds"
                )
            clip = clips[0]
            if int(scalar(clip.input[1], "INT4 clip minimum")) != -8 or int(
                scalar(clip.input[2], "INT4 clip maximum")
            ) != 7:
                raise OnnxImportError(
                    "Q/DQ INT4 Clip bounds must be exactly [-8, 7]"
                )
            quantized = clip.output[0]
        return quantized, scalar(node.input[1], "scale", positive=True)

    builder = GraphBuilder(package_name)
    buffers: dict[str, int] = {}
    input_name = inputs[0].name
    buffers[input_name] = builder.add_buffer(input_name, _shape_for(shapes, input_name), DType.INT8, BufferUsage.ACTIVATION, layout=_layout(_shape_for(shapes, input_name)))

    def activation(name: str, shape_source: str, output: bool = False) -> int:
        if name not in buffers:
            shape = _shape_for(shapes, shape_source)
            buffers[name] = builder.add_buffer(name, shape, DType.INT8, BufferUsage.OUTPUT if output else BufferUsage.ACTIVATION, layout=_layout(shape))
        return buffers[name]

    def weight(name: str, dtype: DType) -> int:
        if name not in buffers:
            array = initializers.get(name)
            if array is None or _array_dtype(array, onnx) != dtype:
                raise OnnxImportError(f"Q/DQ constant {name!r} has the wrong dtype")
            shape = tuple(int(dim) for dim in array.shape) or (1,)
            buffers[name] = builder.add_buffer(name, shape, dtype, BufferUsage.WEIGHT, data=array, layout=_layout(shape))
        return buffers[name]

    graph_output = graph.output[0].name if len(graph.output) == 1 else None
    for node in graph.node:
        if node.op_type not in ("Conv", "Relu", "MaxPool", "AveragePool", "GlobalAveragePool", "Flatten", "Gemm"):
            continue
        q_input, in_scale = qdq_source(node.input[0])
        if q_input not in buffers:
            raise OnnxImportError(f"Q/DQ tensor {q_input!r} is not produced by a supported operator")
        q_output, out_scale = qdq_output(node.output[0])
        is_output = graph_output == q_output or (graph_output in producers and producers[graph_output].op_type == "DequantizeLinear" and producers[graph_output].input[0] == q_output)
        output_idx = activation(q_output, node.output[0], is_output)
        input_idx = buffers[q_input]
        input_shape = _shape_for(shapes, node.input[0])
        output_shape = _shape_for(shapes, node.output[0])
        if node.op_type == "Conv":
            if len(node.input) < 2:
                raise OnnxImportError("Q/DQ Conv requires weights")
            q_weight, weight_scale = qdq_source(node.input[1])
            weight_idx = weight(q_weight, DType.INT8)
            bias_idx = 0
            if len(node.input) > 2:
                q_bias, bias_scale = qdq_source(node.input[2])
                if not math.isclose(bias_scale, in_scale * weight_scale, rel_tol=1e-4, abs_tol=1e-12):
                    raise OnnxImportError("Q/DQ Conv bias scale must equal input_scale * weight_scale")
                bias_idx = weight(q_bias, DType.INT32)
            _require_nchw(input_shape, "Q/DQ Conv input"); _require_nchw(output_shape, "Q/DQ Conv output")
            n, c, h, w = input_shape; _, out_c, out_h, out_w = output_shape
            acc_idx = builder.add_buffer(f"{q_output}.acc", output_shape, DType.INT32, BufferUsage.ACTIVATION, layout=Layout.NCHW)
            builder.conv2d(input_idx, weight_idx, acc_idx, n=n, c=c, h=h, w=w, out_c=out_c, out_h=out_h, out_w=out_w, bias_idx=bias_idx, **_conv_attributes(node, onnx))
            multiplier, right_shift = _fixed_point_multiplier(in_scale * weight_scale / out_scale)
            params = builder.requantize_params(f"{q_output}.requant", multiplier, right_shift)
            builder.requantize(acc_idx, output_idx, params)
        elif node.op_type == "Relu":
            if not math.isclose(in_scale, out_scale, rel_tol=1e-6): raise OnnxImportError("Q/DQ Relu must preserve scale")
            builder.relu(input_idx, output_idx)
        elif node.op_type == "Flatten":
            if not math.isclose(in_scale, out_scale, rel_tol=1e-6): raise OnnxImportError("Q/DQ Flatten must preserve scale")
            if _elements(input_shape) != _elements(output_shape): raise OnnxImportError("Q/DQ Flatten must preserve element count")
            builder.reshape(input_idx, output_idx)
        elif node.op_type == "Gemm":
            if len(node.input) not in (2, 3): raise OnnxImportError("Q/DQ Gemm must have two or three inputs")
            if _attribute(node, "transA", 0, onnx) or _attribute(node, "alpha", 1.0, onnx) != 1.0 or _attribute(node, "beta", 1.0, onnx) != 1.0: raise OnnxImportError("Q/DQ Gemm only supports transA=0, alpha=1, beta=1")
            q_weight, weight_scale = qdq_source(node.input[1])
            if len(input_shape) != 2 or len(output_shape) != 2: raise OnnxImportError("Q/DQ Gemm requires rank-2 input and output")
            m, k = input_shape
            n = output_shape[1]
            trans_b = _attribute(node, "transB", 0, onnx)
            if trans_b == 0:
                if len(node.input) != 2 or _shape_for(shapes, node.input[1]) != (k, n): raise OnnxImportError("Q/DQ Gemm weight shape must be KxN and cannot have bias")
                weight_idx = weight(q_weight, DType.INT8)
                acc_idx = builder.add_buffer(f"{q_output}.acc", output_shape, DType.INT32, BufferUsage.ACTIVATION)
                builder.gemm(input_idx, weight_idx, acc_idx, m=m, n=n, k=k)
                multiplier, right_shift = _fixed_point_multiplier(in_scale * weight_scale / out_scale)
                params = builder.requantize_params(f"{q_output}.requant", multiplier, right_shift)
                builder.requantize(acc_idx, output_idx, params)
            elif trans_b == 1:
                if _shape_for(shapes, node.input[1]) != (n, k): raise OnnxImportError("Q/DQ transposed Gemm weight shape must be NxK")
                weight_array = initializers.get(q_weight)
                if weight_array is None or _array_dtype(weight_array, onnx) != DType.INT8: raise OnnxImportError("Q/DQ transposed Gemm weights must be INT8 initializers")
                conv_weight = builder.add_buffer(f"{q_weight}.conv1x1", (n, k, 1, 1), DType.INT8, BufferUsage.WEIGHT, data=weight_array.reshape(n, k, 1, 1), layout=Layout.NCHW)
                conv_input = builder.add_buffer(f"{q_output}.classifier_input", (m, k, 1, 1), DType.INT8, BufferUsage.ACTIVATION, layout=Layout.NCHW)
                builder.reshape(input_idx, conv_input)
                bias_idx = 0
                if len(node.input) == 3:
                    q_bias, bias_scale = qdq_source(node.input[2])
                    if not math.isclose(bias_scale, in_scale * weight_scale, rel_tol=1e-4, abs_tol=1e-12): raise OnnxImportError("Q/DQ classifier bias scale must equal input_scale * weight_scale")
                    bias_idx = weight(q_bias, DType.INT32)
                    if initializers.get(q_bias) is None or tuple(initializers[q_bias].shape) != (n,): raise OnnxImportError("Q/DQ classifier bias shape must be N")
                conv_acc = builder.add_buffer(f"{q_output}.classifier_acc", (m, n, 1, 1), DType.INT32, BufferUsage.ACTIVATION, layout=Layout.NCHW)
                conv_out = builder.add_buffer(f"{q_output}.classifier_out", (m, n, 1, 1), DType.INT8, BufferUsage.ACTIVATION, layout=Layout.NCHW)
                builder.conv2d(conv_input, conv_weight, conv_acc, n=m, c=k, h=1, w=1, out_c=n, out_h=1, out_w=1, kernel_h=1, kernel_w=1, bias_idx=bias_idx)
                multiplier, right_shift = _fixed_point_multiplier(in_scale * weight_scale / out_scale)
                params = builder.requantize_params(f"{q_output}.requant", multiplier, right_shift)
                builder.requantize(conv_acc, conv_out, params)
                builder.reshape(conv_out, output_idx)
            else:
                raise OnnxImportError("Q/DQ Gemm transB must be 0 or 1")
        else:
            _require_nchw(input_shape, f"Q/DQ {node.op_type} input"); _require_nchw(output_shape, f"Q/DQ {node.op_type} output")
            n, c, h, w = input_shape; _, _, out_h, out_w = output_shape
            if node.op_type == "GlobalAveragePool":
                builder.global_avgpool2d(input_idx, output_idx, n=n, c=c, h=h, w=w)
            else:
                if not math.isclose(in_scale, out_scale, rel_tol=1e-6): raise OnnxImportError(f"Q/DQ {node.op_type} must preserve scale")
                kernel = _attribute(node, "kernel_shape", None, onnx)
                if not kernel or len(kernel) != 2: raise OnnxImportError(f"Q/DQ {node.op_type} requires kernel_shape")
                strides = _attribute(node, "strides", [1, 1], onnx); pads = _pads(_attribute(node, "pads", [0, 0, 0, 0], onnx))
                pool = builder.avgpool2d if node.op_type == "AveragePool" else builder.maxpool2d
                pool(input_idx, output_idx, n=n, c=c, h=h, w=w, out_h=out_h, out_w=out_w, kernel_h=int(kernel[0]), kernel_w=int(kernel[1]), stride_h=int(strides[0]), stride_w=int(strides[1]), pad_h=pads[0], pad_w=pads[1])
    if graph_output is not None and graph_output in producers and producers[graph_output].op_type == "DequantizeLinear": graph_output = producers[graph_output].input[0]
    if graph_output not in buffers: raise OnnxImportError("Q/DQ graph output is not produced by a supported operator")
    return builder.package(buffers[input_name], buffers[graph_output])


def _lower_qlinear_conv(node, builder, buffers, add_initializer, add_activation, initializers, shapes, outputs, onnx):
    if len(node.input) not in (8, 9):
        raise OnnxImportError("QLinearConv must have eight or nine inputs")
    x, x_scale, x_zp, weight, weight_scale, weight_zp, y_scale, y_zp, *bias = node.input
    _require_int8_tensor(x, buffers, builder, "QLinearConv input")
    _require_scalar(initializers, x_scale, "QLinearConv input scale", positive=True)
    _require_scalar_value(initializers, x_zp, 0, "QLinearConv input zero point")
    _require_scalar(initializers, weight_scale, "QLinearConv weight scale", positive=True)
    _require_scalar_value(initializers, weight_zp, 0, "QLinearConv weight zero point")
    out_scale = _require_scalar(initializers, y_scale, "QLinearConv output scale", positive=True)
    out_zp = _require_scalar(initializers, y_zp, "QLinearConv output zero point")
    if not -128 <= out_zp <= 127:
        raise OnnxImportError("QLinearConv output zero point must fit signed INT8")
    weight_idx = add_initializer(weight)
    if builder.buffers[weight_idx].dtype != DType.INT8:
        raise OnnxImportError("QLinearConv weights must be signed INT8")
    input_shape = _shape_for(shapes, x)
    output_shape = _shape_for(shapes, node.output[0])
    _require_nchw(input_shape, "QLinearConv input")
    _require_nchw(output_shape, "QLinearConv output")
    attrs = _conv_attributes(node, onnx)
    bias_idx = 0
    if bias:
        bias_idx = add_initializer(bias[0])
        if builder.buffers[bias_idx].dtype != DType.INT32:
            raise OnnxImportError("QLinearConv bias must be INT32")
    acc_name = f"{node.output[0]}.acc"
    shapes[acc_name] = output_shape
    acc_idx = add_activation(acc_name, DType.INT32)
    n, c, h, w = input_shape
    _, out_c, out_h, out_w = output_shape
    builder.conv2d(buffers[x], weight_idx, acc_idx, n=n, c=c, h=h, w=w, out_c=out_c, out_h=out_h, out_w=out_w, bias_idx=bias_idx, **attrs)
    output_idx = add_activation(node.output[0], DType.INT8, output=node.output[0] in outputs)
    multiplier, right_shift = _fixed_point_multiplier(
        _require_scalar(initializers, x_scale, "QLinearConv input scale", positive=True)
        * _require_scalar(initializers, weight_scale, "QLinearConv weight scale", positive=True)
        / out_scale
    )
    params_idx = builder.requantize_params(f"{node.output[0]}.requant", multiplier, right_shift, out_zp)
    builder.requantize(acc_idx, output_idx, params_idx)


def _lower_conv(node, builder, buffers, add_initializer, add_activation, initializers, shapes, outputs, onnx):
    if len(node.input) not in (2, 3):
        raise OnnxImportError("Conv must have two or three inputs")
    x, weight, *bias = node.input
    _require_int8_tensor(x, buffers, builder, "Conv input")
    weight_idx = add_initializer(weight)
    if builder.buffers[weight_idx].dtype != DType.INT8:
        raise OnnxImportError("Conv weights must be INT8")
    bias_idx = add_initializer(bias[0]) if bias else 0
    if bias and builder.buffers[bias_idx].dtype != DType.INT32:
        raise OnnxImportError("Conv bias must be INT32")
    input_shape, output_shape = _shape_for(shapes, x), _shape_for(shapes, node.output[0])
    _require_nchw(input_shape, "Conv input")
    _require_nchw(output_shape, "Conv output")
    output_idx = add_activation(node.output[0], DType.INT32, output=node.output[0] in outputs)
    n, c, h, w = input_shape
    _, out_c, out_h, out_w = output_shape
    builder.conv2d(buffers[x], weight_idx, output_idx, n=n, c=c, h=h, w=w, out_c=out_c, out_h=out_h, out_w=out_w, bias_idx=bias_idx, **_conv_attributes(node, onnx))


def _lower_activation(node, builder, buffers, add_activation, shapes, outputs, onnx):
    x = node.input[0]
    _require_known_tensor(x, buffers, node.op_type)
    dtype = builder.buffers[buffers[x]].dtype
    output_idx = add_activation(node.output[0], dtype, output=node.output[0] in outputs)
    if node.op_type == "Relu":
        builder.relu(buffers[x], output_idx)
        return
    alpha = _attribute(node, "alpha", 0.01, onnx)
    if alpha <= 0 or alpha > 1:
        raise OnnxImportError("LeakyRelu alpha must be a positive power of two no larger than one")
    shift = round(-math.log2(alpha))
    if shift > 7 or not math.isclose(alpha, 2.0 ** -shift, rel_tol=0.0, abs_tol=1e-8):
        raise OnnxImportError("LeakyRelu alpha must equal 2^-N for N in 0..7")
    builder.leaky_relu(buffers[x], output_idx, shift)


def _lower_maxpool(node, builder, buffers, add_activation, shapes, outputs, onnx):
    x = node.input[0]
    op_name = node.op_type
    _require_known_tensor(x, buffers, op_name)
    if builder.buffers[buffers[x]].dtype != DType.INT8:
        raise OnnxImportError(f"{op_name} input must be INT8")
    input_shape, output_shape = _shape_for(shapes, x), _shape_for(shapes, node.output[0])
    _require_nchw(input_shape, f"{op_name} input")
    _require_nchw(output_shape, f"{op_name} output")
    n, c, h, w = input_shape
    _, _, out_h, out_w = output_shape
    output_idx = add_activation(node.output[0], DType.INT8, output=node.output[0] in outputs)
    if op_name == "GlobalAveragePool":
        if (out_h, out_w) != (1, 1):
            raise OnnxImportError("GlobalAveragePool output must be 1x1")
        builder.global_avgpool2d(buffers[x], output_idx, n=n, c=c, h=h, w=w)
        return
    kernel = _attribute(node, "kernel_shape", None, onnx)
    if not kernel or len(kernel) != 2:
        raise OnnxImportError(f"{op_name} requires a two-dimensional kernel_shape")
    strides = _attribute(node, "strides", [1, 1], onnx)
    pads = _pads(_attribute(node, "pads", [0, 0, 0, 0], onnx))
    if _attribute(node, "ceil_mode", 0, onnx) or _attribute(node, "dilations", [1, 1], onnx) != [1, 1]:
        raise OnnxImportError(f"{op_name} ceil_mode and dilation are not supported")
    pool = builder.avgpool2d if op_name == "AveragePool" else builder.maxpool2d
    pool(buffers[x], output_idx, n=n, c=c, h=h, w=w, out_h=out_h, out_w=out_w, kernel_h=int(kernel[0]), kernel_w=int(kernel[1]), stride_h=int(strides[0]), stride_w=int(strides[1]), pad_h=pads[0], pad_w=pads[1])


def _lower_reshape(node, builder, buffers, add_activation, shapes, outputs, onnx):
    x = node.input[0]
    _require_known_tensor(x, buffers, node.op_type)
    src_shape, output_shape = _shape_for(shapes, x), _shape_for(shapes, node.output[0])
    if _elements(src_shape) != _elements(output_shape):
        raise OnnxImportError(f"{node.op_type} must preserve the element count")
    dtype = builder.buffers[buffers[x]].dtype
    output_idx = add_activation(node.output[0], dtype, output=node.output[0] in outputs)
    builder.reshape(buffers[x], output_idx)


def _lower_gemm(node, builder, buffers, add_initializer, add_activation, initializers, shapes, outputs, onnx):
    if len(node.input) < 2 or len(node.input) > 3:
        raise OnnxImportError(f"{node.op_type} must have two or three inputs")
    x, weight, *bias = node.input
    _require_int8_tensor(x, buffers, builder, f"{node.op_type} input")
    if bias:
        raise OnnxImportError(f"{node.op_type} bias is not implemented; use a preceding Conv bias or a separate supported path")
    weight_idx = add_initializer(weight)
    if builder.buffers[weight_idx].dtype != DType.INT8:
        raise OnnxImportError(f"{node.op_type} weights must be INT8")
    x_shape, w_shape, output_shape = _shape_for(shapes, x), _shape_for(shapes, weight), _shape_for(shapes, node.output[0])
    if len(x_shape) != 2 or len(w_shape) != 2 or len(output_shape) != 2:
        raise OnnxImportError(f"{node.op_type} currently supports rank-2 matrices only")
    if node.op_type == "Gemm":
        if _attribute(node, "transA", 0, onnx) != 0 or _attribute(node, "alpha", 1.0, onnx) != 1.0 or _attribute(node, "beta", 1.0, onnx) != 1.0:
            raise OnnxImportError("Gemm only supports transA=0, alpha=1, beta=1")
        if _attribute(node, "transB", 0, onnx) != 0:
            raise OnnxImportError("Gemm with transB=1 is not yet supported; export MatMul with KxN weights")
    m, k = x_shape
    if w_shape[0] != k or output_shape != (m, w_shape[1]):
        raise OnnxImportError(f"{node.op_type} matrix dimensions are incompatible")
    output_idx = add_activation(node.output[0], DType.INT32, output=node.output[0] in outputs)
    builder.layers.append(Layer(OpType.GEMM, input0=buffers[x], output=output_idx, weight=weight_idx, n=m, c=k, out_c=w_shape[1]))


def _require_onnx():
    try:
        import onnx
        from onnx import numpy_helper
    except ModuleNotFoundError as exc:
        raise OnnxImportError("ONNX import support requires the optional dependency 'onnx'. Install it with: python -m pip install onnx") from exc
    return onnx, numpy_helper


def _collect_shapes(graph, overrides):
    shapes = {}
    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                dims.append(int(dim.dim_value))
            else:
                dims = []
                break
        if dims:
            shapes[value.name] = tuple(dims)
    for tensor in graph.initializer:
        shapes[tensor.name] = tuple(int(dim) for dim in tensor.dims) or (1,)
    for tensor_name, shape in overrides.items():
        if not shape or any(int(dim) <= 0 for dim in shape):
            raise OnnxImportError(f"Input override for {tensor_name!r} must use positive dimensions")
        shapes[tensor_name] = tuple(int(dim) for dim in shape)
    return shapes


def _shape_for(shapes, tensor_name):
    try:
        return shapes[tensor_name]
    except KeyError as exc:
        raise OnnxImportError(f"Tensor {tensor_name!r} has no static inferred shape") from exc


def _onnx_dtype(elem_type, onnx):
    if elem_type == onnx.TensorProto.INT8:
        return DType.INT8
    if elem_type == onnx.TensorProto.INT32:
        return DType.INT32
    raise OnnxImportError(f"Unsupported ONNX tensor type {elem_type}; only INT8 and INT32 are deployable")


def _array_dtype(array, onnx):
    return _onnx_dtype(onnx.helper.np_dtype_to_tensor_dtype(array.dtype), onnx)


def _layout(shape):
    return Layout.NCHW if len(shape) == 4 else Layout.ROW_MAJOR


def _require_known_tensor(name, buffers, context):
    if name not in buffers:
        raise OnnxImportError(f"{context} input {name!r} is not a supported prior activation")


def _require_int8_tensor(name, buffers, builder, context):
    _require_known_tensor(name, buffers, context)
    if builder.buffers[buffers[name]].dtype != DType.INT8:
        raise OnnxImportError(f"{context} must be signed INT8")


def _require_nchw(shape, context):
    if len(shape) != 4:
        raise OnnxImportError(f"{context} must use static NCHW rank-4 tensors")


def _attribute(node, name, default, onnx):
    for attribute in node.attribute:
        if attribute.name == name:
            value = onnx.helper.get_attribute_value(attribute)
            return list(value) if isinstance(value, tuple) else value
    return default


def _pads(pads):
    if len(pads) == 2:
        return int(pads[0]), int(pads[1])
    if len(pads) != 4 or pads[0] != pads[2] or pads[1] != pads[3]:
        raise OnnxImportError("Only symmetric 2D padding is supported")
    return int(pads[0]), int(pads[1])


def _conv_attributes(node, onnx):
    if _attribute(node, "auto_pad", "NOTSET", onnx) not in ("NOTSET", b"NOTSET"):
        raise OnnxImportError("Conv auto_pad is not supported; export explicit symmetric pads")
    if _attribute(node, "group", 1, onnx) != 1:
        raise OnnxImportError("Grouped convolution is not supported")
    dilations = _attribute(node, "dilations", [1, 1], onnx)
    if list(dilations) != [1, 1]:
        raise OnnxImportError("Dilated convolution is not supported")
    strides = _attribute(node, "strides", [1, 1], onnx)
    kernel = _attribute(node, "kernel_shape", None, onnx)
    if len(strides) != 2 or not kernel or len(kernel) != 2:
        raise OnnxImportError("Conv requires two-dimensional kernel_shape and strides")
    pad_h, pad_w = _pads(_attribute(node, "pads", [0, 0, 0, 0], onnx))
    return {"kernel_h": int(kernel[0]), "kernel_w": int(kernel[1]), "stride_h": int(strides[0]), "stride_w": int(strides[1]), "pad_h": pad_h, "pad_w": pad_w}


def _require_scalar(initializers, name, label, positive=False):
    try:
        values = initializers[name].reshape(-1).tolist()
    except KeyError as exc:
        raise OnnxImportError(f"{label} must be a scalar initializer") from exc
    if len(values) != 1:
        raise OnnxImportError(f"{label} must be per-tensor (one scalar), not per-channel")
    value = float(values[0])
    if not math.isfinite(value) or (positive and value <= 0):
        raise OnnxImportError(f"{label} must be {'positive ' if positive else ''}finite")
    return value


def _require_scalar_value(initializers, name, expected, label):
    value = _require_scalar(initializers, name, label)
    if value != expected:
        raise OnnxImportError(f"{label} must be {expected}; this datapath has no zero-point subtraction")
    return int(value)


def _fixed_point_multiplier(scale):
    if not math.isfinite(scale) or scale <= 0:
        raise OnnxImportError("Requantization scale must be positive and finite")
    for right_shift in range(30, -1, -1):
        multiplier = int(round(scale * (1 << right_shift)))
        if 0 < multiplier <= 0x7FFFFFFF:
            return multiplier, right_shift
    raise OnnxImportError("Requantization scale cannot be represented by the INT32 fixed-point requantizer")


def _elements(shape):
    total = 1
    for dim in shape:
        total *= int(dim)
    return total
