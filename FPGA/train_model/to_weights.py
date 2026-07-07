from pathlib import Path
import torch
import numpy as np

from paths import FLOAT_CHECKPOINT, HLS_EXPORT_DIR


# ============================================================
# User settings
# ============================================================

CHECKPOINT_PATH = FLOAT_CHECKPOINT
# Alternative:
# Use export_qat_quant_params.py for QAT/INT8 exports.

OUT_DIR = HLS_EXPORT_DIR
OUT_WEIGHTS = OUT_DIR / "weights.h"

# These are initial conservative fixed-point choices.
# You will tune these after comparing accuracy.
WEIGHT_SCALE = 64.0

CONV1_SHIFT = 8
CONV2_SHIFT = 10
CONV3_SHIFT = 10
FC_SHIFT = 8


# ============================================================
# Helpers
# ============================================================

def get_state_dict(ckpt):
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def get_tensor(state, possible_names):
    for name in possible_names:
        if name in state:
            return state[name].detach().cpu().float()
    raise KeyError(f"Could not find any of: {possible_names}")


def fold_conv_bn(conv_w, conv_b, bn_w, bn_b, bn_mean, bn_var, bn_eps=1e-5):
    """
    Folds BatchNorm into Conv.

    conv output:
        y = conv(x, W, b)

    BN:
        z = gamma * (y - mean) / sqrt(var + eps) + beta

    folded:
        z = conv(x, W_folded, b_folded)
    """

    if conv_b is None:
        conv_b = torch.zeros(conv_w.shape[0])

    scale = bn_w / torch.sqrt(bn_var + bn_eps)

    w_folded = conv_w * scale.reshape(-1, 1, 1, 1)
    b_folded = bn_b + (conv_b - bn_mean) * scale

    return w_folded, b_folded


def quantize_weight_int8(w, scale):
    q = torch.round(w * scale)
    q = torch.clamp(q, -128, 127)
    return q.to(torch.int8)


def quantize_bias_int32(b, scale):
    q = torch.round(b * scale)
    return q.to(torch.int32)


def tensor_to_cpp_array(tensor, name, ctype):
    arr = tensor.cpu().numpy()
    shape = arr.shape

    def recurse(a, indent=0):
        spaces = " " * indent

        if a.ndim == 1:
            values = ", ".join(str(int(x)) for x in a)
            return "{ " + values + " }"

        inner = []
        for sub in a:
            inner.append(recurse(sub, indent + 4))

        return "{\n" + ",\n".join((" " * (indent + 4)) + x for x in inner) + "\n" + spaces + "}"

    shape_str = "".join(f"[{d}]" for d in shape)
    return f"static const {ctype} {name}{shape_str} = {recurse(arr)};\n"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state = get_state_dict(ckpt)

    # ------------------------------------------------------------
    # Support both naming styles.
    # ------------------------------------------------------------

    conv1_w = get_tensor(state, ["conv1.weight", "features.0.weight"])
    bn1_w = get_tensor(state, ["bn1.weight", "features.1.weight"])
    bn1_b = get_tensor(state, ["bn1.bias", "features.1.bias"])
    bn1_mean = get_tensor(state, ["bn1.running_mean", "features.1.running_mean"])
    bn1_var = get_tensor(state, ["bn1.running_var", "features.1.running_var"])

    conv2_w = get_tensor(state, ["conv2.weight", "features.4.weight"])
    bn2_w = get_tensor(state, ["bn2.weight", "features.5.weight"])
    bn2_b = get_tensor(state, ["bn2.bias", "features.5.bias"])
    bn2_mean = get_tensor(state, ["bn2.running_mean", "features.5.running_mean"])
    bn2_var = get_tensor(state, ["bn2.running_var", "features.5.running_var"])

    conv3_w = get_tensor(state, ["conv3.weight", "features.8.weight"])
    bn3_w = get_tensor(state, ["bn3.weight", "features.9.weight"])
    bn3_b = get_tensor(state, ["bn3.bias", "features.9.bias"])
    bn3_mean = get_tensor(state, ["bn3.running_mean", "features.9.running_mean"])
    bn3_var = get_tensor(state, ["bn3.running_var", "features.9.running_var"])

    fc_w = get_tensor(state, ["fc.weight", "classifier.weight"])
    fc_b = get_tensor(state, ["fc.bias", "classifier.bias"])

    # ------------------------------------------------------------
    # Fold BatchNorm into Conv.
    # Original convs used bias=False, so conv_b=None.
    # ------------------------------------------------------------

    conv1_w_f, conv1_b_f = fold_conv_bn(
        conv1_w, None, bn1_w, bn1_b, bn1_mean, bn1_var
    )

    conv2_w_f, conv2_b_f = fold_conv_bn(
        conv2_w, None, bn2_w, bn2_b, bn2_mean, bn2_var
    )

    conv3_w_f, conv3_b_f = fold_conv_bn(
        conv3_w, None, bn3_w, bn3_b, bn3_mean, bn3_var
    )

    # ------------------------------------------------------------
    # Quantize.
    #
    # First pass:
    #   weights scaled by WEIGHT_SCALE
    #   folded biases scaled by WEIGHT_SCALE
    #
    # The layer right-shifts compensate the accumulated scale.
    # This is not perfect calibration yet, but it is testable.
    # ------------------------------------------------------------

    conv1_w_q = quantize_weight_int8(conv1_w_f, WEIGHT_SCALE)
    conv2_w_q = quantize_weight_int8(conv2_w_f, WEIGHT_SCALE)
    conv3_w_q = quantize_weight_int8(conv3_w_f, WEIGHT_SCALE)
    fc_w_q = quantize_weight_int8(fc_w, WEIGHT_SCALE)

    conv1_b_q = quantize_bias_int32(conv1_b_f, WEIGHT_SCALE)
    conv2_b_q = quantize_bias_int32(conv2_b_f, WEIGHT_SCALE)
    conv3_b_q = quantize_bias_int32(conv3_b_f, WEIGHT_SCALE)
    fc_b_q = quantize_bias_int32(fc_b, WEIGHT_SCALE)

    # ------------------------------------------------------------
    # Emit weights.h.
    # ------------------------------------------------------------

    header = []
    header.append("#ifndef FRUIT_CNN_WEIGHTS_H\n")
    header.append("#define FRUIT_CNN_WEIGHTS_H\n\n")
    header.append('#include "test_cnn.h"\n\n')

    header.append("// Auto-generated by to_weights.py\n")
    header.append("// Do not hand-edit.\n\n")

    header.append(f"#define CONV1_SHIFT {CONV1_SHIFT}\n")
    header.append(f"#define CONV2_SHIFT {CONV2_SHIFT}\n")
    header.append(f"#define CONV3_SHIFT {CONV3_SHIFT}\n")
    header.append(f"#define FC_SHIFT    {FC_SHIFT}\n\n")

    header.append(tensor_to_cpp_array(conv1_w_q, "conv1_w", "weight_t"))
    header.append(tensor_to_cpp_array(conv1_b_q, "conv1_b", "bias_t"))

    header.append(tensor_to_cpp_array(conv2_w_q, "conv2_w", "weight_t"))
    header.append(tensor_to_cpp_array(conv2_b_q, "conv2_b", "bias_t"))

    header.append(tensor_to_cpp_array(conv3_w_q, "conv3_w", "weight_t"))
    header.append(tensor_to_cpp_array(conv3_b_q, "conv3_b", "bias_t"))

    header.append(tensor_to_cpp_array(fc_w_q, "fc_w", "weight_t"))
    header.append(tensor_to_cpp_array(fc_b_q, "fc_b", "bias_t"))

    header.append("\n#endif\n")

    OUT_WEIGHTS.write_text("".join(header), encoding="utf-8")

    print(f"Wrote: {OUT_WEIGHTS}")
    print("Quantization summary:")
    print(f"  WEIGHT_SCALE = {WEIGHT_SCALE}")
    print(f"  CONV1_SHIFT  = {CONV1_SHIFT}")
    print(f"  CONV2_SHIFT  = {CONV2_SHIFT}")
    print(f"  CONV3_SHIFT  = {CONV3_SHIFT}")
    print(f"  FC_SHIFT     = {FC_SHIFT}")

    print()
    print("Weight ranges:")
    print("  conv1:", int(conv1_w_q.min()), int(conv1_w_q.max()))
    print("  conv2:", int(conv2_w_q.min()), int(conv2_w_q.max()))
    print("  conv3:", int(conv3_w_q.min()), int(conv3_w_q.max()))
    print("  fc:   ", int(fc_w_q.min()), int(fc_w_q.max()))


if __name__ == "__main__":
    main()
