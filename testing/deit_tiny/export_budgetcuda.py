"""Export the trained six-class DeiT-Tiny checkpoint for Budget-CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FPGA"))
from budgetcuda.packing import pack_transformer_weights

CLASSES = ("apple_fresh", "apple_rotten", "banana_fresh", "banana_rotten", "orange_fresh", "orange_rotten")


def quantize_symmetric(tensor: torch.Tensor, bits: int) -> tuple[torch.Tensor, float]:
    limit = (1 << (bits - 1)) - 1
    scale = max(float(tensor.detach().abs().max()) / limit, 1e-12)
    values = torch.round(tensor.detach() / scale).clamp(-limit - 1, limit).to(torch.int8)
    return values, scale


def fixed_multiplier(real_scale: float) -> tuple[int, int]:
    shift = 30
    multiplier = int(round(real_scale * (1 << shift)))
    while abs(multiplier) > 0x7FFFFFFF and shift:
        shift -= 1
        multiplier = int(round(real_scale * (1 << shift)))
    if abs(multiplier) > 0x7FFFFFFF:
        raise ValueError(f"Scale {real_scale} cannot be represented as signed Q multiplier")
    return multiplier, shift


def tensor_scale(state: dict[str, torch.Tensor], name: str, default: float = 1.0) -> float:
    value = state.get(name)
    return default if value is None else max(float(value.reshape(-1)[0]), 1e-12)


def export_linear(
    state, prefix: str, output_dir: Path, input_scale: float, output_scale: float,
    manifest: dict, weight_bits: int,
) -> None:
    weight = state[f"{prefix}.weight"].detach().cpu()
    if weight.ndim != 2:
        raise ValueError(f"{prefix}.weight is not linear")
    packed_weight, weight_scale = quantize_symmetric(weight, weight_bits)
    k_by_n = packed_weight.t().contiguous().tolist()
    int4 = weight_bits == 4
    raw = pack_transformer_weights(k_by_n, weight.shape[1], weight.shape[0], int4=int4)
    filename = prefix.replace(".", "_") + f".w{weight_bits}.bin"
    (output_dir / filename).write_bytes(raw)
    bias = state.get(f"{prefix}.bias")
    bias_int32 = [] if bias is None else torch.round(bias.detach().cpu() / (input_scale * weight_scale)).to(torch.int32).tolist()
    multiplier, right_shift = fixed_multiplier(input_scale * weight_scale / output_scale)
    manifest["tensors"][prefix] = {
        "file": filename, "sha256": hashlib.sha256(raw).hexdigest(),
        "shape_k_n": [weight.shape[1], weight.shape[0]],
        "layout": f"transformer_weight_tiled_int{weight_bits}",
        "weight_scale": weight_scale, "input_scale": input_scale, "output_scale": output_scale,
        "requant": [multiplier, right_shift, 0, 0], "bias_int32": bias_int32,
    }


def export_checkpoint(checkpoint_path: Path, output_dir: Path, weight_bits: int = 8) -> Path:
    if weight_bits not in (4, 8):
        raise ValueError("weight_bits must be 4 or 8")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if tuple(checkpoint.get("class_names", ())) != CLASSES:
        raise ValueError("Checkpoint class order is not the six-class fruit contract")
    state = checkpoint["state_dict"]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": f"budgetcuda-deit-w{weight_bits}a8", "version": 1,
        "weight_bits": weight_bits, "activation_bits": 8, "model": checkpoint.get("model_name"),
        "classes": list(CLASSES), "image_size": 224, "patch_size": 16, "tokens": 197,
        "hidden_dim": 192, "heads": 3, "head_dim": 64, "depth": 12, "mlp_dim": 768,
        "tensors": {}, "blocks": [],
    }

    patch_weight = state["patch_embed.proj.weight"].reshape(192, -1)
    patch_state = dict(state); patch_state["patch_embed.linear.weight"] = patch_weight
    patch_state["patch_embed.linear.bias"] = state["patch_embed.proj.bias"]
    patch_in = tensor_scale(state, "patch_embed.proj.input_quant.scale", 1 / 128)
    patch_out = tensor_scale(state, "patch_embed.proj.output_quant.scale", 1 / 128)
    export_linear(patch_state, "patch_embed.linear", output_dir, patch_in, patch_out, manifest, weight_bits)

    cls_pos = torch.cat((state["cls_token"], torch.zeros(1, 196, 192)), dim=1) + state["pos_embed"]
    cls_pos_q, cls_pos_scale = quantize_symmetric(cls_pos, 8)
    constants = cls_pos_q.numpy().tobytes()
    (output_dir / "class_position_tokens.i8.bin").write_bytes(constants)
    manifest["class_position_tokens"] = {"file": "class_position_tokens.i8.bin", "scale": cls_pos_scale, "shape": [197, 192]}

    for block in range(12):
        base = f"blocks.{block}"
        qkv = state[f"{base}.attn.qkv.weight"]
        qkv_bias = state[f"{base}.attn.qkv.bias"]
        qkv_in = tensor_scale(state, f"{base}.attn.qkv.input_quant.scale")
        qkv_out = tensor_scale(state, f"{base}.attn.qkv.output_quant.scale")
        for index, label in enumerate(("q", "k", "v")):
            alias = f"{base}.attn.{label}"
            state[alias + ".weight"] = qkv[index * 192:(index + 1) * 192]
            state[alias + ".bias"] = qkv_bias[index * 192:(index + 1) * 192]
            export_linear(state, alias, output_dir, qkv_in, qkv_out, manifest, weight_bits)
        proj_in = tensor_scale(state, f"{base}.attn.proj.input_quant.scale")
        proj_out = tensor_scale(state, f"{base}.attn.proj.output_quant.scale")
        export_linear(state, f"{base}.attn.proj", output_dir, proj_in, proj_out, manifest, weight_bits)
        fc1_in = tensor_scale(state, f"{base}.mlp.fc1.input_quant.scale")
        fc1_out = tensor_scale(state, f"{base}.mlp.fc1.output_quant.scale")
        fc2_in = tensor_scale(state, f"{base}.mlp.fc2.input_quant.scale")
        fc2_out = tensor_scale(state, f"{base}.mlp.fc2.output_quant.scale")
        export_linear(state, f"{base}.mlp.fc1", output_dir, fc1_in, fc1_out, manifest, weight_bits)
        export_linear(state, f"{base}.mlp.fc2", output_dir, fc2_in, fc2_out, manifest, weight_bits)
        manifest["blocks"].append({
            "index": block, "sequence": ["layernorm1", "qkv", "qk_transpose", "qk_gemm", "softmax", "av_gemm", "attention_projection", "residual1", "layernorm2", "fc1", "gelu", "fc2", "residual2"],
            "norm1_gamma": state[f"{base}.norm1.weight"].tolist(), "norm1_beta": state[f"{base}.norm1.bias"].tolist(),
            "norm2_gamma": state[f"{base}.norm2.weight"].tolist(), "norm2_beta": state[f"{base}.norm2.bias"].tolist(),
        })
    head_in = tensor_scale(state, "head.input_quant.scale", 1.0)
    head_out = tensor_scale(state, "head.output_quant.scale", 1.0)
    export_linear(state, "head", output_dir, head_in, head_out, manifest, weight_bits)
    manifest["final_norm"] = {"gamma": state["norm.weight"].tolist(), "beta": state["norm.bias"].tolist()}
    manifest["execution"] = ["patch_embedding", "prepend_class_and_add_position", *[f"encoder_block_{i}" for i in range(12)], "final_layernorm", "class_token_select", "six_class_head"]
    output = output_dir / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument("--checkpoint", type=Path, default=here / "artifacts" / "w4a8" / "best_fake_quant.pt")
    parser.add_argument("--output-dir", type=Path, default=here / "artifacts" / "budgetcuda_w8a8")
    parser.add_argument("--weight-bits", type=int, choices=(4, 8), default=8)
    args = parser.parse_args()
    print(export_checkpoint(args.checkpoint, args.output_dir, args.weight_bits))


if __name__ == "__main__":
    main()
