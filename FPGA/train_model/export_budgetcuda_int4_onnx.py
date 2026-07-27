"""Export a BudgetCUDA INT4-QAT checkpoint as a canonical Q/DQ ONNX model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from paths import INT4_ONNX, INT4_QAT_CHECKPOINT
from qat_model import CLASS_NAMES, QATFruitCNN64, prepare_budgetcuda_qat


def _scalar(name, value, dtype):
    return numpy_helper.from_array(np.asarray(value, dtype=dtype), name)


def _fake_quant_scale(module) -> float:
    scale = float(module.scale.reshape(-1)[0])
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("QAT observer contains an invalid scale")
    return scale


def _quantized_weight(module) -> tuple[np.ndarray, float]:
    scale = _fake_quant_scale(module.weight_fake_quant)
    values = torch.trunc(module.weight.detach().cpu() / scale).clamp(-8, 7)
    return values.to(torch.int8).numpy(), scale


def export_int4_qdq(model: torch.nn.Module, output_path: Path) -> None:
    model = model.cpu().eval()
    initializers = []
    nodes = []

    def add_qparams(
        prefix: str, scale: float, zero_point_dtype=np.int8
    ) -> tuple[str, str]:
        scale_name = f"{prefix}.scale"
        zp_name = f"{prefix}.zero_point"
        initializers.extend(
            [
                _scalar(scale_name, scale, np.float32),
                _scalar(zp_name, 0, zero_point_dtype),
            ]
        )
        return scale_name, zp_name

    def dequant(
        value: str, prefix: str, scale: float, zero_point_dtype=np.int8
    ) -> str:
        scale_name, zp_name = add_qparams(prefix, scale, zero_point_dtype)
        output = f"{prefix}.dequant"
        nodes.append(helper.make_node("DequantizeLinear", [value, scale_name, zp_name], [output]))
        return output

    def quantize_int4(
        value: str, prefix: str, scale: float
    ) -> tuple[str, str, str]:
        scale_name, zp_name = add_qparams(prefix, scale)
        raw_quantized = f"{prefix}.quantized_raw"
        quantized = f"{prefix}.quantized"
        min_name = f"{prefix}.quant_min"
        max_name = f"{prefix}.quant_max"
        initializers.extend(
            [
                _scalar(min_name, -8, np.int8),
                _scalar(max_name, 7, np.int8),
            ]
        )
        nodes.append(
            helper.make_node(
                "QuantizeLinear",
                [value, scale_name, zp_name],
                [raw_quantized],
            )
        )
        nodes.append(
            helper.make_node(
                "Clip",
                [raw_quantized, min_name, max_name],
                [quantized],
            )
        )
        return quantized, scale_name, zp_name

    def quant_dequant(value: str, prefix: str, scale: float) -> tuple[str, str]:
        quantized, scale_name, zp_name = quantize_int4(value, prefix, scale)
        dequantized = f"{prefix}.dequant"
        nodes.append(
            helper.make_node(
                "DequantizeLinear", [quantized, scale_name, zp_name], [dequantized]
            )
        )
        return quantized, dequantized

    input_scale = _fake_quant_scale(model.quant.activation_post_process)
    current = dequant("input", "input", input_scale)

    for index, (conv, pool) in enumerate(
        ((model.conv1, model.pool1), (model.conv2, model.pool2), (model.conv3, None)),
        start=1,
    ):
        weight, weight_scale = _quantized_weight(conv)
        weight_name = f"conv{index}.weight"
        initializers.append(numpy_helper.from_array(weight, weight_name))
        weight_dq = dequant(weight_name, f"conv{index}.weight_qdq", weight_scale)

        conv_inputs = [current, weight_dq]
        if conv.bias is not None:
            bias_scale = input_scale * weight_scale
            bias = torch.round(conv.bias.detach().cpu() / bias_scale).to(torch.int32).numpy()
            bias_name = f"conv{index}.bias"
            initializers.append(numpy_helper.from_array(bias, bias_name))
            conv_inputs.append(
                dequant(
                    bias_name,
                    f"conv{index}.bias_qdq",
                    bias_scale,
                    np.int32,
                )
            )

        conv_float = f"conv{index}.float"
        nodes.append(
            helper.make_node(
                "Conv",
                conv_inputs,
                [conv_float],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
                strides=[1, 1],
            )
        )
        output_scale = _fake_quant_scale(conv.activation_post_process)
        _, conv_dq = quant_dequant(conv_float, f"conv{index}.output", output_scale)
        relu_float = f"relu{index}.float"
        nodes.append(helper.make_node("Relu", [conv_dq], [relu_float]))
        _, current = quant_dequant(relu_float, f"relu{index}.output", output_scale)
        if pool is not None:
            pool_float = f"pool{index}.float"
            nodes.append(
                helper.make_node(
                    "MaxPool",
                    [current],
                    [pool_float],
                    kernel_shape=[2, 2],
                    strides=[2, 2],
                )
            )
            _, current = quant_dequant(pool_float, f"pool{index}.output", output_scale)
        input_scale = output_scale

    avg_float = "avgpool.float"
    nodes.append(
        helper.make_node(
            "AveragePool",
            [current],
            [avg_float],
            kernel_shape=[16, 16],
            strides=[16, 16],
        )
    )
    _, current = quant_dequant(avg_float, "avgpool.output", input_scale)
    flatten_float = "flatten.float"
    nodes.append(helper.make_node("Flatten", [current], [flatten_float], axis=1))
    _, current = quant_dequant(flatten_float, "flatten.output", input_scale)

    weight, weight_scale = _quantized_weight(model.fc)
    initializers.append(numpy_helper.from_array(weight, "fc.weight"))
    weight_dq = dequant("fc.weight", "fc.weight_qdq", weight_scale)
    bias_scale = input_scale * weight_scale
    bias = torch.round(model.fc.bias.detach().cpu() / bias_scale).to(torch.int32).numpy()
    initializers.append(numpy_helper.from_array(bias, "fc.bias"))
    bias_dq = dequant("fc.bias", "fc.bias_qdq", bias_scale, np.int32)
    nodes.append(
        helper.make_node(
            "Gemm", [current, weight_dq, bias_dq], ["fc.float"], transB=1
        )
    )
    output_scale = _fake_quant_scale(model.fc.activation_post_process)
    output_quantized, output_scale_name, output_zp_name = quantize_int4(
        "fc.float", "output", output_scale
    )
    nodes.append(
        helper.make_node(
            "DequantizeLinear",
            [output_quantized, output_scale_name, output_zp_name],
            ["output"],
        )
    )

    graph = helper.make_graph(
        nodes,
        "fruit3_budgetcuda_int4",
        [helper.make_tensor_value_info("input", TensorProto.INT8, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
        initializers,
    )
    onnx_model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
        producer_name="budgetcuda",
    )
    onnx.checker.check_model(onnx_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onnx_model, output_path)


def load_qat_checkpoint(path: Path) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    channels = tuple(checkpoint.get("channels", (12, 24, 48)))
    model = prepare_budgetcuda_qat(QATFruitCNN64(len(CLASS_NAMES), channels=channels))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.eval(), checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=INT4_QAT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=INT4_ONNX)
    args = parser.parse_args()

    model, checkpoint = load_qat_checkpoint(args.checkpoint)
    export_int4_qdq(model, args.output)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from budgetcuda.onnx_import import import_onnx_model

    package = import_onnx_model(args.output)
    manifest = {
        "model": str(args.output.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "class_names": checkpoint.get("class_names", CLASS_NAMES),
        "channels": checkpoint.get("channels", [12, 24, 48]),
        "input_shape": list(package.buffers[package.input_index].shape),
        "input_dtype": package.buffers[package.input_index].dtype.name.lower(),
        "output_dtype": package.buffers[package.output_index].dtype.name.lower(),
        "layers": len(package.layers),
        "buffers": len(package.buffers),
        "quant_min": -8,
        "quant_max": 7,
        "container_dtype": "int8",
        "deployment_dtype": "packed_int4",
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
