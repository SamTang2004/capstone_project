from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.ao.quantization as tq

from qat_model import QATFruitCNN64, select_quant_backend
from paths import HLS_EXPORT_DIR, INT8_FULL_MODEL, INT8_STATE, QAT_CHECKPOINT, QUANT_EXPORT_DIR


DEFAULT_QAT_CHECKPOINT = QAT_CHECKPOINT
DEFAULT_INT8_STATE = INT8_STATE
DEFAULT_INT8_FULL_MODEL = INT8_FULL_MODEL
DEFAULT_JSON_DIR = QUANT_EXPORT_DIR
DEFAULT_HLS_DIR = HLS_EXPORT_DIR
CLASS_NAMES = ["apple", "banana", "orange"]
REQUANT_SHIFT = 24


def _to_float_list(tensor: torch.Tensor) -> list[float]:
    return [float(x) for x in tensor.detach().cpu().reshape(-1).tolist()]


def _to_int_list(tensor: torch.Tensor) -> list[int]:
    return [int(x) for x in tensor.detach().cpu().reshape(-1).tolist()]


def _tensor_to_cpp_array(tensor: torch.Tensor, name: str, ctype: str) -> str:
    arr = tensor.detach().cpu().numpy()

    def recurse(a, indent: int = 0) -> str:
        spaces = " " * indent
        if a.ndim == 1:
            return "{ " + ", ".join(str(int(x)) for x in a) + " }"

        inner = []
        for sub in a:
            inner.append(recurse(sub, indent + 4))
        return "{\n" + ",\n".join((" " * (indent + 4)) + x for x in inner) + "\n" + spaces + "}"

    shape_str = "".join(f"[{d}]" for d in arr.shape)
    return f"static const {ctype} {name}{shape_str} = {recurse(arr)};\n"


def _list_to_cpp_array(values: list[int], name: str, ctype: str) -> str:
    joined = ", ".join(str(int(v)) for v in values)
    return f"static const {ctype} {name}[{len(values)}] = {{ {joined} }};\n"


def _requant_multipliers(scales: torch.Tensor) -> torch.Tensor:
    mult = torch.round(scales.to(torch.float64) * float(1 << REQUANT_SHIFT)).to(torch.int64)
    mult = torch.clamp(mult, 0, 2**31 - 1)
    return mult.to(torch.int32)


def load_converted_model_from_qat(checkpoint_path: Path) -> torch.nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"QAT checkpoint not found: {checkpoint_path}")

    qengine = select_quant_backend()
    torch.backends.quantized.engine = qengine

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_fp32 = QATFruitCNN64(num_classes=3)
    model_fp32.eval()
    model_fp32.fuse_model()
    model_fp32.train()
    model_fp32.qconfig = tq.get_default_qat_qconfig(qengine)
    model_qat = tq.prepare_qat(model_fp32, inplace=False)

    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    model_qat.load_state_dict(state, strict=False)
    model_qat = model_qat.cpu()
    model_qat.eval()

    model_int8 = tq.convert(model_qat, inplace=False)
    model_int8.eval()
    return model_int8


def build_empty_converted_model(qengine: str) -> torch.nn.Module:
    torch.backends.quantized.engine = qengine

    model_fp32 = QATFruitCNN64(num_classes=3)
    model_fp32.eval()
    model_fp32.fuse_model()
    model_fp32.train()
    model_fp32.qconfig = tq.get_default_qat_qconfig(qengine)
    model_qat = tq.prepare_qat(model_fp32, inplace=False)
    model_qat = model_qat.cpu()
    model_qat.eval()

    model_int8 = tq.convert(model_qat, inplace=False)
    model_int8.eval()
    return model_int8


def load_converted_model_from_state(model_path: Path) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"INT8 state checkpoint not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Expected INT8 state checkpoint with model_state_dict: {model_path}")

    qengine = checkpoint.get("qengine") or select_quant_backend()
    model = build_empty_converted_model(qengine)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def load_converted_model_from_full_model(model_path: Path) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"INT8 full model not found: {model_path}")

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def module_weight_qtensor(module: torch.nn.Module) -> torch.Tensor:
    if hasattr(module, "weight"):
        weight_attr = module.weight
        if callable(weight_attr):
            return weight_attr()
        if isinstance(weight_attr, torch.Tensor):
            return weight_attr
    raise TypeError(f"Could not access quantized weight tensor from module {module}")


def module_bias_tensor(module: torch.nn.Module) -> torch.Tensor:
    if hasattr(module, "bias"):
        bias_attr = module.bias
        if callable(bias_attr):
            bias = bias_attr()
            if bias is not None:
                return bias.detach().cpu()
        elif bias_attr is not None:
            return bias_attr.detach().cpu()
    raise TypeError(f"Could not access bias tensor from module {module}")


def extract_quant_stub(model: torch.nn.Module) -> dict:
    return {
        "scale": float(model.quant.scale),
        "zero_point": int(model.quant.zero_point),
        "dtype": "quint8",
    }


def extract_quantized_layer(
    module: torch.nn.Module,
    name: str,
    input_scale: float,
    input_zero_point: int,
) -> dict:
    weight_q = module_weight_qtensor(module)
    bias_fp32 = module_bias_tensor(module)

    if weight_q.qscheme() not in {torch.per_channel_affine, torch.per_channel_affine_float_qparams}:
        raise ValueError(f"{name}: unsupported qscheme {weight_q.qscheme()}")

    weight_int = weight_q.int_repr().to(torch.int32).cpu()
    weight_scales = weight_q.q_per_channel_scales().to(torch.float64).cpu()
    weight_zero_points = weight_q.q_per_channel_zero_points().to(torch.int64).cpu()

    output_scale = float(module.scale)
    output_zero_point = int(module.zero_point)

    bias_int32 = torch.round(
        bias_fp32.to(torch.float64) / (float(input_scale) * weight_scales)
    ).to(torch.int32)

    requant_scales = (float(input_scale) * weight_scales) / float(output_scale)
    requant_mult = _requant_multipliers(requant_scales)

    return {
        "name": name,
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "weight_qscheme": str(weight_q.qscheme()),
        "weight_shape": list(weight_q.shape),
        "weight_scales": _to_float_list(weight_scales),
        "weight_zero_points": _to_int_list(weight_zero_points),
        "weight_int_repr": weight_int.tolist(),
        "bias_fp32": _to_float_list(bias_fp32),
        "bias_int32": _to_int_list(bias_int32),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
        "requant_scales": _to_float_list(requant_scales),
        "requant_mult": _to_int_list(requant_mult),
        "requant_shift": REQUANT_SHIFT,
    }


def emit_quant_params_header(export: dict, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("#ifndef FRUIT_CNN_QUANT_PARAMS_H\n")
    lines.append("#define FRUIT_CNN_QUANT_PARAMS_H\n\n")
    lines.append('#include "test_cnn.h"\n\n')
    lines.append("// Auto-generated by export_qat_quant_params.py\n")
    lines.append("// These parameters mirror PyTorch quantization metadata.\n\n")

    input_quant = export["input_quant"]
    lines.append(f"static const int INPUT_ZERO_POINT = {input_quant['zero_point']};\n\n")

    for layer_name in ("conv1", "conv2", "conv3", "fc"):
        layer = export["layers"][layer_name]
        macro = layer_name.upper()
        lines.append(f"#define {macro}_REQUANT_SHIFT {layer['requant_shift']}\n")
        lines.append(f"static const int {layer_name}_input_zero_point = {layer['input_zero_point']};\n")
        lines.append(f"static const int {layer_name}_output_zero_point = {layer['output_zero_point']};\n")
        lines.append(_list_to_cpp_array(layer["weight_zero_points"], f"{layer_name}_weight_zero_point", "int"))
        lines.append(_list_to_cpp_array(layer["requant_mult"], f"{layer_name}_requant_mult", "int"))
        lines.append("\n")

    lines.append("#endif\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def emit_weights_header(export: dict, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("#ifndef FRUIT_CNN_WEIGHTS_H\n")
    lines.append("#define FRUIT_CNN_WEIGHTS_H\n\n")
    lines.append('#include "test_cnn.h"\n\n')
    lines.append("// Auto-generated by export_qat_quant_params.py\n")
    lines.append("// Weights and biases come from a converted PyTorch INT8 model.\n\n")

    conv1 = export["layers"]["conv1"]
    conv2 = export["layers"]["conv2"]
    conv3 = export["layers"]["conv3"]
    fc = export["layers"]["fc"]

    lines.append(_tensor_to_cpp_array(torch.tensor(conv1["weight_int_repr"], dtype=torch.int32), "conv1_w", "weight_t"))
    lines.append(_tensor_to_cpp_array(torch.tensor(conv1["bias_int32"], dtype=torch.int32), "conv1_b", "bias_t"))
    lines.append(_tensor_to_cpp_array(torch.tensor(conv2["weight_int_repr"], dtype=torch.int32), "conv2_w", "weight_t"))
    lines.append(_tensor_to_cpp_array(torch.tensor(conv2["bias_int32"], dtype=torch.int32), "conv2_b", "bias_t"))
    lines.append(_tensor_to_cpp_array(torch.tensor(conv3["weight_int_repr"], dtype=torch.int32), "conv3_w", "weight_t"))
    lines.append(_tensor_to_cpp_array(torch.tensor(conv3["bias_int32"], dtype=torch.int32), "conv3_b", "bias_t"))
    lines.append(_tensor_to_cpp_array(torch.tensor(fc["weight_int_repr"], dtype=torch.int32), "fc_w", "weight_t"))
    lines.append(_tensor_to_cpp_array(torch.tensor(fc["bias_int32"], dtype=torch.int32), "fc_b", "bias_t"))
    lines.append("\n#endif\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def build_export(model: torch.nn.Module, source_path: Path) -> dict:
    input_quant = extract_quant_stub(model)

    conv1 = extract_quantized_layer(model.conv1, "conv1", input_quant["scale"], input_quant["zero_point"])
    conv2 = extract_quantized_layer(model.conv2, "conv2", conv1["output_scale"], conv1["output_zero_point"])
    conv3 = extract_quantized_layer(model.conv3, "conv3", conv2["output_scale"], conv2["output_zero_point"])
    fc = extract_quantized_layer(model.fc, "fc", conv3["output_scale"], conv3["output_zero_point"])

    return {
        "source": str(source_path),
        "class_names": CLASS_NAMES,
        "input_quant": input_quant,
        "layers": {
            "conv1": conv1,
            "conv2": conv2,
            "conv3": conv3,
            "fc": fc,
        },
        "notes": [
            "Weights are exported from PyTorch converted INT8 modules.",
            "bias_int32[channel] = round(bias_fp32 / (input_scale * weight_scale[channel])).",
            "requant(acc) ~= round(acc * requant_mult / 2^REQUANT_SHIFT) + output_zero_point.",
            "The current HLS path still uses uint8 activations and per-channel requantization.",
            "AvgPool is assumed to preserve conv3 quantization parameters for the FC input path.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export PyTorch QAT/INT8 weights and quantization parameters for HLS."
    )
    parser.add_argument("--qat-checkpoint", type=Path, default=DEFAULT_QAT_CHECKPOINT)
    parser.add_argument("--int8-state", type=Path, default=DEFAULT_INT8_STATE)
    parser.add_argument("--int8-full-model", type=Path, default=DEFAULT_INT8_FULL_MODEL)
    parser.add_argument(
        "--source",
        choices=["int8_state", "int8_full_model", "qat_checkpoint"],
        default="int8_state",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=DEFAULT_JSON_DIR,
        help="Directory for exported JSON diagnostics.",
    )
    parser.add_argument(
        "--hls-dir",
        type=Path,
        default=DEFAULT_HLS_DIR,
        help="Directory where weights.h and quant_params.h should be written.",
    )
    args = parser.parse_args()

    args.json_dir.mkdir(parents=True, exist_ok=True)
    args.hls_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "int8_state":
        model = load_converted_model_from_state(args.int8_state)
        source_path = args.int8_state
    elif args.source == "int8_full_model":
        model = load_converted_model_from_full_model(args.int8_full_model)
        source_path = args.int8_full_model
    else:
        model = load_converted_model_from_qat(args.qat_checkpoint)
        source_path = args.qat_checkpoint

    export = build_export(model, source_path)

    json_path = args.json_dir / "qat_quant_params.json"
    weights_path = args.hls_dir / "weights.h"
    quant_path = args.hls_dir / "quant_params.h"

    json_path.write_text(json.dumps(export, indent=2), encoding="utf-8")
    emit_weights_header(export, weights_path)
    emit_quant_params_header(export, quant_path)

    print(f"Exported diagnostic JSON: {json_path}")
    print(f"Exported HLS weights:     {weights_path}")
    print(f"Exported HLS quant:       {quant_path}")
    print("Layer scales:")
    for name in ("conv1", "conv2", "conv3", "fc"):
        layer = export["layers"][name]
        print(
            f"  {name}: in_scale={layer['input_scale']:.8f}, "
            f"out_scale={layer['output_scale']:.8f}, "
            f"out_zp={layer['output_zero_point']}, "
            f"requant_shift={layer['requant_shift']}"
        )


if __name__ == "__main__":
    main()
