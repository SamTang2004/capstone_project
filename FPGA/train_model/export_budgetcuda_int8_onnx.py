"""Export the signed BudgetCUDA QAT fruit model as canonical INT8 Q/DQ ONNX.

PyTorch's stock ONNX QAT export uses unsigned image Q/DQ plus conversion nodes.
That graph is correct for a CPU runtime but cannot run on BudgetCUDA's signed,
zero-centred PL datapath.  This exporter writes the small supported ONNX subset
directly from a checkpoint trained with ``train_budgetcuda_qat.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from paths import MODELS_DIR
from qat_model import CLASS_NAMES, IMAGE_SIZE, QATFruitCNN64, prepare_budgetcuda_qat


DEFAULT_DIR = MODELS_DIR / "fruit3_budgetcuda_qat_smoke"
DEFAULT_CHECKPOINT = DEFAULT_DIR / "best_budgetcuda_qat_fakequant.pt"
DEFAULT_OUTPUT = DEFAULT_DIR / "fruit3_budgetcuda_int8.onnx"


def _scalar(module) -> float:
    value = module.scale.detach().cpu().reshape(-1)
    if value.numel() != 1 or float(value[0]) <= 0:
        raise ValueError("BudgetCUDA export requires positive per-tensor fake-quant scales")
    return float(value[0])


def _quantize(values: torch.Tensor, scale: float, dtype: np.dtype) -> np.ndarray:
    limit = 127 if dtype == np.int8 else np.iinfo(dtype).max
    array = torch.round(values.detach().cpu() / scale).clamp(-limit - (dtype == np.int8), limit)
    return array.to(torch.int8 if dtype == np.int8 else torch.int32).numpy()


def load_model(checkpoint: Path) -> tuple[torch.nn.Module, dict]:
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    profile = checkpoint_data.get("quantization", "")
    if "signed qint8" not in profile or "zero point 0" not in profile:
        raise ValueError("checkpoint is not a signed symmetric BudgetCUDA QAT checkpoint")
    model = prepare_budgetcuda_qat(QATFruitCNN64(num_classes=len(CLASS_NAMES)))
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()
    return model, checkpoint_data


def export_model(model: torch.nn.Module, output: Path) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    def initializer(name: str, array: np.ndarray):
        return numpy_helper.from_array(np.asarray(array), name=name)

    initializers = []
    nodes = []
    scales: dict[str, float] = {
        "input": _scalar(model.quant.activation_post_process),
        "conv1": _scalar(model.conv1.activation_post_process),
        "conv2": _scalar(model.conv2.activation_post_process),
        "conv3": _scalar(model.conv3.activation_post_process),
        "fc": _scalar(model.fc.activation_post_process),
    }

    for name, value in scales.items():
        initializers.extend((initializer(f"{name}_scale", np.array(value, dtype=np.float32)), initializer(f"{name}_zp", np.array(0, dtype=np.int8))))

    def dq(q_name: str, scale_name: str, out_name: str) -> None:
        nodes.append(helper.make_node("DequantizeLinear", [q_name, f"{scale_name}_scale", f"{scale_name}_zp"], [out_name]))

    def q(value_name: str, scale_name: str, out_name: str) -> None:
        nodes.append(helper.make_node("QuantizeLinear", [value_name, f"{scale_name}_scale", f"{scale_name}_zp"], [out_name]))

    def add_weight(prefix: str, layer, input_scale: float) -> tuple[str, str]:
        weight_scale = _scalar(layer.weight_fake_quant)
        weight_name, bias_name = f"{prefix}_weight", f"{prefix}_bias"
        initializers.extend((
            initializer(weight_name, _quantize(layer.weight, weight_scale, np.int8)),
            initializer(f"{prefix}_weight_scale", np.array(weight_scale, dtype=np.float32)),
            initializer(f"{prefix}_weight_zp", np.array(0, dtype=np.int8)),
            initializer(bias_name, _quantize(layer.bias, input_scale * weight_scale, np.int32)),
            initializer(f"{prefix}_bias_scale", np.array(input_scale * weight_scale, dtype=np.float32)),
            initializer(f"{prefix}_bias_zp", np.array(0, dtype=np.int32)),
        ))
        return weight_name, bias_name

    def conv(prefix: str, layer, q_input: str, input_scale_name: str, output_scale_name: str) -> str:
        input_scale = scales[input_scale_name]
        weight_name, bias_name = add_weight(prefix, layer, input_scale)
        dq(q_input, input_scale_name, f"{prefix}_input")
        dq(weight_name, f"{prefix}_weight", f"{prefix}_weight_dq")
        dq(bias_name, f"{prefix}_bias", f"{prefix}_bias_dq")
        nodes.append(helper.make_node("Conv", [f"{prefix}_input", f"{prefix}_weight_dq", f"{prefix}_bias_dq"], [f"{prefix}_conv"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]))
        q(f"{prefix}_conv", output_scale_name, f"q_{prefix}_conv")
        dq(f"q_{prefix}_conv", output_scale_name, f"{prefix}_relu_input")
        nodes.append(helper.make_node("Relu", [f"{prefix}_relu_input"], [f"{prefix}_relu"]))
        q(f"{prefix}_relu", output_scale_name, f"q_{prefix}_relu")
        return f"q_{prefix}_relu"

    def maxpool(prefix: str, q_input: str, scale_name: str) -> str:
        dq(q_input, scale_name, f"{prefix}_input")
        nodes.append(helper.make_node("MaxPool", [f"{prefix}_input"], [f"{prefix}_pool"], kernel_shape=[2, 2], strides=[2, 2]))
        q(f"{prefix}_pool", scale_name, f"q_{prefix}")
        return f"q_{prefix}"

    q_value = conv("conv1", model.conv1, "input", "input", "conv1")
    q_value = maxpool("pool1", q_value, "conv1")
    q_value = conv("conv2", model.conv2, q_value, "conv1", "conv2")
    q_value = maxpool("pool2", q_value, "conv2")
    q_value = conv("conv3", model.conv3, q_value, "conv2", "conv3")
    dq(q_value, "conv3", "avgpool_input")
    nodes.append(helper.make_node("AveragePool", ["avgpool_input"], ["avgpool"], kernel_shape=[16, 16], strides=[16, 16]))
    q("avgpool", "conv3", "q_avgpool")
    dq("q_avgpool", "conv3", "flatten_input")
    nodes.append(helper.make_node("Flatten", ["flatten_input"], ["flatten"], axis=1))
    q("flatten", "conv3", "q_flatten")

    weight_name, bias_name = add_weight("fc", model.fc, scales["conv3"])
    dq("q_flatten", "conv3", "fc_input")
    dq(weight_name, "fc_weight", "fc_weight_dq")
    dq(bias_name, "fc_bias", "fc_bias_dq")
    nodes.append(helper.make_node("Gemm", ["fc_input", "fc_weight_dq", "fc_bias_dq"], ["fc"], transB=1))
    q("fc", "fc", "q_logits")
    dq("q_logits", "fc", "logits")

    graph = helper.make_graph(
        nodes,
        "fruit3_budgetcuda_signed_int8",
        [helper.make_tensor_value_info("input", TensorProto.INT8, [1, 3, IMAGE_SIZE, IMAGE_SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, len(CLASS_NAMES)])],
        initializers,
    )
    model_proto = helper.make_model(graph, producer_name="BudgetCUDA", opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model_proto)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_proto, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    model, metadata = load_model(args.checkpoint)
    export_model(model, args.output)
    manifest = {
        "onnx": str(args.output), "checkpoint": str(args.checkpoint), "input_name": "input",
        "input_dtype": "int8", "input_shape": [1, 3, IMAGE_SIZE, IMAGE_SIZE],
        "output_name": "logits", "class_names": metadata.get("class_names", CLASS_NAMES),
        "quantization": "signed int8, per-tensor symmetric, zero point 0",
        "requantization": "BudgetCUDA PL fixed-point truncation toward zero; ONNX QuantizeLinear is a graph interchange reference only",
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported signed BudgetCUDA ONNX: {args.output}")


if __name__ == "__main__":
    main()
