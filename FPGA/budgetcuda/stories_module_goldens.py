"""Integer-observable W4A8 module simulators for TinyStories.

These primitives preserve the current host fake-quant contract while exposing
the integer codes, scales, INT32 accumulators, and dequantized values needed by
firmware and RTL module tests. Signed symmetric quantization intentionally uses
[-qmax, qmax], so W4 codes occupy [-7, 7] even though the packed hardware lane
can represent -8.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .stories import Llama2CCheckpoint, Llama2CTokenizer
from .stories_quantization import StoriesFakeQuantReference, StoriesQuantizationSpec


@dataclass(frozen=True)
class QuantizedValue:
    codes: np.ndarray
    scale: np.ndarray
    dequantized: np.ndarray


@dataclass(frozen=True)
class QuantizedWeight:
    codes: np.ndarray
    scales: np.ndarray


def quantize_symmetric(
    value: np.ndarray, bits: int, *, axes=None,
) -> QuantizedValue:
    value = np.asarray(value, dtype=np.float32)
    qmax = np.float32((1 << (bits - 1)) - 1)
    maximum = np.max(np.abs(value), axis=axes, keepdims=axes is not None)
    scale = np.asarray(maximum / qmax, dtype=np.float32)
    scale = np.where(scale == 0, np.float32(1.0), scale).astype(np.float32)
    codes = np.clip(np.rint(value / scale), -qmax, qmax).astype(np.int8)
    dequantized = np.asarray(codes.astype(np.float32) * scale, dtype=np.float32)
    return QuantizedValue(codes, scale, dequantized)


def quantize_probability(value: np.ndarray, bits: int = 8) -> QuantizedValue:
    value = np.asarray(value, dtype=np.float32)
    qmax = np.float32((1 << bits) - 1)
    codes = np.rint(np.clip(value, 0.0, 1.0) * qmax).astype(np.uint8)
    scale = np.asarray(np.float32(1.0) / qmax, dtype=np.float32)
    return QuantizedValue(
        codes, scale, np.asarray(codes.astype(np.float32) * scale, dtype=np.float32),
    )


def quantize_weight_per_output(value: np.ndarray, bits: int = 4) -> QuantizedWeight:
    quantized = quantize_symmetric(value, bits, axes=-1)
    return QuantizedWeight(quantized.codes, quantized.scale)


def pack_int4_tiled(weight_codes: np.ndarray) -> np.ndarray:
    """Pack logical [output, input] codes as accelerator [N-tile][K][4]."""

    codes = np.asarray(weight_codes, dtype=np.int8)
    if codes.ndim != 2 or np.any(codes < -8) or np.any(codes > 7):
        raise ValueError("INT4 weight codes must have shape [output,input] and be in [-8,7]")
    output, reduction = codes.shape
    padded_output = (output + 7) & ~7
    matrix = np.zeros((reduction, padded_output), dtype=np.int8)
    matrix[:, :output] = codes.T
    lanes = matrix.reshape(reduction, padded_output // 8, 8).transpose(1, 0, 2)
    low = lanes[..., 0::2].astype(np.uint8) & np.uint8(0x0F)
    high = (lanes[..., 1::2].astype(np.uint8) & np.uint8(0x0F)) << np.uint8(4)
    return np.ascontiguousarray((low | high).reshape(-1), dtype=np.uint8)


def pack_int8_activation_vector(codes: np.ndarray) -> np.ndarray:
    """Pack one logical activation row as [one M-tile][K][8 row lanes]."""

    codes = np.asarray(codes, dtype=np.int8)
    if codes.ndim != 1:
        raise ValueError("activation codes must be a vector")
    packed = np.zeros((codes.size, 8), dtype=np.int8)
    packed[:, 0] = codes
    return np.ascontiguousarray(packed.view(np.uint8).reshape(-1))


def w4a8_linear(
    value: np.ndarray,
    weight: QuantizedWeight,
    *,
    quantize_output: bool = True,
    capture: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Run one vector through W4A8 GEMM with explicit INT32 accumulation."""

    activation = quantize_symmetric(value, 8)
    codes = np.asarray(weight.codes, dtype=np.int8)
    if codes.ndim != 2 or activation.codes.shape != (codes.shape[1],):
        raise ValueError("linear input and weight reduction dimensions differ")
    accumulator = codes.astype(np.int32) @ activation.codes.astype(np.int32)
    output_scale = np.asarray(
        weight.scales.reshape(-1) * activation.scale.reshape(()), dtype=np.float32,
    )
    pre_activation = np.asarray(accumulator.astype(np.float32) * output_scale, dtype=np.float32)
    tensors = {
        "input_float": np.asarray(value, dtype=np.float32),
        "input_q": activation.codes,
        "input_scale": activation.scale,
        "input_dq": activation.dequantized,
        "weight_q": codes,
        "weight_scale_per_output": np.asarray(weight.scales, dtype=np.float32).reshape(-1),
        "accumulator_int32": np.asarray(accumulator, dtype=np.int32),
        "accumulator_scale_per_output": output_scale,
        "pre_activation_float": pre_activation,
    }
    if capture:
        tensors["input_packed_tiled_int8"] = pack_int8_activation_vector(activation.codes)
        tensors["weight_packed_tiled_int4"] = pack_int4_tiled(codes)
    if not quantize_output:
        tensors["output_float"] = pre_activation
        return pre_activation, tensors
    output = quantize_symmetric(pre_activation, 8)
    tensors.update({
        "output_q": output.codes,
        "output_scale": output.scale,
        "output_dq": output.dequantized,
    })
    return output.dequantized, tensors


def embedding_gather(
    token: int, weight: QuantizedWeight,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if token < 0 or token >= weight.codes.shape[0]:
        raise ValueError("embedding token is outside the vocabulary")
    codes = np.asarray(weight.codes[token], dtype=np.int8)
    scale = np.asarray(weight.scales[token], dtype=np.float32)
    dequantized = np.asarray(codes.astype(np.float32) * scale, dtype=np.float32)
    output = quantize_symmetric(dequantized, 8)
    return output.dequantized, {
        "token": np.asarray(token, dtype=np.int32),
        "weight_q": codes,
        "weight_scale": scale,
        "weight_dq": dequantized,
        "output_q": output.codes,
        "output_scale": output.scale,
        "output_dq": output.dequantized,
    }


def rmsnorm(
    value: np.ndarray, weight: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    value = np.asarray(value, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    mean_square = np.asarray(np.mean(value * value, dtype=np.float32), dtype=np.float32)
    inverse_rms = np.asarray(
        np.float32(1.0) / np.sqrt(mean_square + np.float32(1e-5), dtype=np.float32),
        dtype=np.float32,
    )
    float_output = np.asarray(weight * (value * inverse_rms), dtype=np.float32)
    output = quantize_symmetric(float_output, 8)
    return output.dequantized, {
        "input_dq": value,
        "weight_float": weight,
        "mean_square_float": mean_square,
        "inverse_rms_float": inverse_rms,
        "pre_activation_float": float_output,
        "output_q": output.codes,
        "output_scale": output.scale,
        "output_dq": output.dequantized,
    }


def rope(
    q: np.ndarray, k: np.ndarray, position: int, *, n_heads: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    q_float = np.asarray(q, dtype=np.float32).copy()
    k_float = np.asarray(k, dtype=np.float32).copy()
    head_size = q_float.size // n_heads
    pair_index = np.arange(0, head_size, 2, dtype=np.float32)
    frequency = np.asarray(1.0 / (10000.0 ** (pair_index / head_size)), dtype=np.float32)
    angle = np.asarray(np.float32(position) * frequency, dtype=np.float32)
    cosine = np.cos(angle, dtype=np.float32)
    sine = np.sin(angle, dtype=np.float32)

    def rotate(value: np.ndarray) -> np.ndarray:
        shaped = value.reshape(-1, head_size)
        first = shaped[:, 0::2].copy()
        second = shaped[:, 1::2].copy()
        shaped[:, 0::2] = first * cosine - second * sine
        shaped[:, 1::2] = first * sine + second * cosine
        return value

    rotate(q_float)
    rotate(k_float)
    q_output = quantize_symmetric(q_float, 8)
    k_output = quantize_symmetric(k_float, 8)
    return q_output.dequantized, k_output.dequantized, {
        "position": np.asarray(position, dtype=np.int32),
        "q_input_dq": np.asarray(q, dtype=np.float32),
        "k_input_dq": np.asarray(k, dtype=np.float32),
        "frequency_float": frequency,
        "cosine_float": cosine,
        "sine_float": sine,
        "q_rotated_float": q_float,
        "k_rotated_float": k_float,
        "q_output_q": q_output.codes,
        "q_output_scale": q_output.scale,
        "q_output_dq": q_output.dequantized,
        "k_output_q": k_output.codes,
        "k_output_scale": k_output.scale,
        "k_output_dq": k_output.dequantized,
    }


def causal_attention(
    q: np.ndarray,
    key_cache: np.ndarray,
    value_cache: np.ndarray,
    *,
    n_heads: int,
    n_kv_heads: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    q = np.asarray(q, dtype=np.float32)
    keys = np.asarray(key_cache, dtype=np.float32)
    values = np.asarray(value_cache, dtype=np.float32)
    head_size = q.size // n_heads
    if keys.shape != values.shape or keys.ndim != 2:
        raise ValueError("K/V caches must have the same [time,kv_dim] shape")
    if keys.shape[1] != head_size * n_kv_heads:
        raise ValueError("K/V cache width does not match the head configuration")

    logits_float = np.empty((n_heads, keys.shape[0]), dtype=np.float32)
    logits_q = np.empty_like(logits_float, dtype=np.int8)
    logits_scale = np.empty(n_heads, dtype=np.float32)
    probability_float = np.empty_like(logits_float)
    probability_q = np.empty_like(logits_float, dtype=np.uint8)
    probability_dq = np.empty_like(logits_float)
    head_float = np.empty((n_heads, head_size), dtype=np.float32)
    head_q = np.empty_like(head_float, dtype=np.int8)
    head_scale = np.empty(n_heads, dtype=np.float32)
    head_dq = np.empty_like(head_float)
    kv_multiplier = n_heads // n_kv_heads
    for head in range(n_heads):
        q_head = q[head * head_size:(head + 1) * head_size]
        kv_head = head // kv_multiplier
        start = kv_head * head_size
        key_head = keys[:, start:start + head_size]
        scores = np.asarray((key_head @ q_head) / np.float32(math.sqrt(head_size)), dtype=np.float32)
        score_quant = quantize_symmetric(scores, 8)
        shifted = np.asarray(score_quant.dequantized - np.max(score_quant.dequantized), dtype=np.float32)
        exponent = np.exp(shifted, dtype=np.float32)
        probabilities = np.asarray(exponent / np.sum(exponent, dtype=np.float32), dtype=np.float32)
        prob_quant = quantize_probability(probabilities, 8)
        value_head = values[:, start:start + head_size]
        context = np.asarray(prob_quant.dequantized @ value_head, dtype=np.float32)
        context_quant = quantize_symmetric(context, 8)
        logits_float[head] = scores
        logits_q[head] = score_quant.codes
        logits_scale[head] = score_quant.scale
        probability_float[head] = probabilities
        probability_q[head] = prob_quant.codes
        probability_dq[head] = prob_quant.dequantized
        head_float[head] = context
        head_q[head] = context_quant.codes
        head_scale[head] = context_quant.scale
        head_dq[head] = context_quant.dequantized
    concatenated = head_dq.reshape(-1)
    output = quantize_symmetric(concatenated, 8)
    return output.dequantized, {
        "q_dq": q,
        "key_cache_dq": keys,
        "value_cache_dq": values,
        "attention_logits_float": logits_float,
        "attention_logits_q": logits_q,
        "attention_logits_scale_per_head": logits_scale,
        "softmax_probability_float": probability_float,
        "softmax_probability_q": probability_q,
        "softmax_probability_scale": np.asarray(np.float32(1.0 / 255.0), dtype=np.float32),
        "softmax_probability_dq": probability_dq,
        "attention_head_float": head_float,
        "attention_head_q": head_q,
        "attention_head_scale": head_scale,
        "attention_head_dq": head_dq,
        "output_q": output.codes,
        "output_scale": output.scale,
        "output_dq": output.dequantized,
    }


def residual_add(
    left: np.ndarray, right: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    summed = np.asarray(np.asarray(left, dtype=np.float32) + np.asarray(right, dtype=np.float32), dtype=np.float32)
    output = quantize_symmetric(summed, 8)
    return output.dequantized, {
        "left_dq": np.asarray(left, dtype=np.float32),
        "right_dq": np.asarray(right, dtype=np.float32),
        "sum_float": summed,
        "output_q": output.codes,
        "output_scale": output.scale,
        "output_dq": output.dequantized,
    }


def silu(value: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    value = np.asarray(value, dtype=np.float32)
    float_output = np.asarray(value / (np.float32(1.0) + np.exp(-value, dtype=np.float32)), dtype=np.float32)
    output = quantize_symmetric(float_output, 8)
    return output.dequantized, {
        "input_dq": value,
        "pre_activation_float": float_output,
        "output_q": output.codes,
        "output_scale": output.scale,
        "output_dq": output.dequantized,
    }


def swiglu(
    silu_gate: np.ndarray, up: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    multiplied = np.asarray(
        np.asarray(silu_gate, dtype=np.float32) * np.asarray(up, dtype=np.float32),
        dtype=np.float32,
    )
    output = quantize_symmetric(multiplied, 8)
    return output.dequantized, {
        "silu_gate_dq": np.asarray(silu_gate, dtype=np.float32),
        "up_dq": np.asarray(up, dtype=np.float32),
        "product_float": multiplied,
        "output_q": output.codes,
        "output_scale": output.scale,
        "output_dq": output.dequantized,
    }


class StoriesW4A8ModuleSimulator:
    """Sequential W4A8 decoder assembled from independently testable modules."""

    def __init__(self, checkpoint: Llama2CCheckpoint):
        self.checkpoint = checkpoint
        self.config = checkpoint.config
        self.embedding = quantize_weight_per_output(checkpoint.token_embedding_table, 4)
        self.weights = {
            name: tuple(quantize_weight_per_output(layer, 4) for layer in getattr(checkpoint, name))
            for name in ("wq", "wk", "wv", "wo", "w1", "w2", "w3")
        }
        c = self.config
        self.key_cache = np.zeros((c.n_layers, c.seq_len, c.kv_dim), dtype=np.float32)
        self.value_cache = np.zeros_like(self.key_cache)
        self.next_position = 0

    def reset(self) -> None:
        self.key_cache.fill(0)
        self.value_cache.fill(0)
        self.next_position = 0

    def forward(
        self, token: int, position: int | None = None, *, capture: bool = False,
    ) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
        c = self.config
        if position is None:
            position = self.next_position
        if position != self.next_position:
            raise ValueError(f"module simulator expects position {self.next_position}, got {position}")
        if position >= c.seq_len:
            raise ValueError("position exceeds checkpoint context")
        modules: dict[str, dict[str, np.ndarray]] = {}

        def keep(name: str, tensors: dict[str, np.ndarray]) -> None:
            if capture:
                modules[name] = tensors

        x, tensors = embedding_gather(token, self.embedding)
        keep("embedding_gather", tensors)
        for layer in range(c.n_layers):
            prefix = f"layer_{layer:02d}"
            normalized, tensors = rmsnorm(x, self.checkpoint.rms_att_weight[layer])
            keep(f"{prefix}/rms_attention", tensors)
            q, tensors = w4a8_linear(
                normalized, self.weights["wq"][layer], capture=capture,
            )
            keep(f"{prefix}/q_projection", tensors)
            k, tensors = w4a8_linear(
                normalized, self.weights["wk"][layer], capture=capture,
            )
            keep(f"{prefix}/k_projection", tensors)
            v, tensors = w4a8_linear(
                normalized, self.weights["wv"][layer], capture=capture,
            )
            keep(f"{prefix}/v_projection", tensors)
            q, k, tensors = rope(q, k, position, n_heads=c.n_heads)
            keep(f"{prefix}/rope", tensors)

            self.key_cache[layer, position] = k
            self.value_cache[layer, position] = v
            if capture:
                k_quant = quantize_symmetric(k, 8)
                v_quant = quantize_symmetric(v, 8)
                keep(f"{prefix}/kv_cache", {
                    "position": np.asarray(position, dtype=np.int32),
                    "key_write_q": k_quant.codes,
                    "key_write_scale": k_quant.scale,
                    "key_write_dq": k,
                    "value_write_q": v_quant.codes,
                    "value_write_scale": v_quant.scale,
                    "value_write_dq": v,
                    "key_cache_prefix_dq": self.key_cache[layer, :position + 1].copy(),
                    "value_cache_prefix_dq": self.value_cache[layer, :position + 1].copy(),
                })
            context, tensors = causal_attention(
                q, self.key_cache[layer, :position + 1],
                self.value_cache[layer, :position + 1],
                n_heads=c.n_heads, n_kv_heads=c.n_kv_heads,
            )
            keep(f"{prefix}/causal_attention", tensors)
            attention_output, tensors = w4a8_linear(
                context, self.weights["wo"][layer], capture=capture,
            )
            keep(f"{prefix}/attention_output_projection", tensors)
            x, tensors = residual_add(x, attention_output)
            keep(f"{prefix}/post_attention_residual", tensors)

            normalized, tensors = rmsnorm(x, self.checkpoint.rms_ffn_weight[layer])
            keep(f"{prefix}/rms_ffn", tensors)
            gate, tensors = w4a8_linear(
                normalized, self.weights["w1"][layer], capture=capture,
            )
            keep(f"{prefix}/ffn_gate_projection", tensors)
            up, tensors = w4a8_linear(
                normalized, self.weights["w3"][layer], capture=capture,
            )
            keep(f"{prefix}/ffn_up_projection", tensors)
            activated_gate, tensors = silu(gate)
            keep(f"{prefix}/silu", tensors)
            gated, tensors = swiglu(activated_gate, up)
            keep(f"{prefix}/swiglu", tensors)
            ffn_output, tensors = w4a8_linear(
                gated, self.weights["w2"][layer], capture=capture,
            )
            keep(f"{prefix}/ffn_down_projection", tensors)
            x, tensors = residual_add(x, ffn_output)
            keep(f"{prefix}/block_residual", tensors)

        normalized, tensors = rmsnorm(x, self.checkpoint.rms_final_weight)
        keep("final_rmsnorm", tensors)
        logits, tensors = w4a8_linear(
            normalized, self.embedding, quantize_output=False, capture=capture,
        )
        keep("vocabulary_projection", tensors)
        self.next_position += 1
        return logits, modules

    def run_tokens(
        self, tokens: Sequence[int], *, capture_position: int | None = None,
    ) -> tuple[tuple[np.ndarray, ...], dict[str, dict[str, np.ndarray]]]:
        if not tokens:
            raise ValueError("at least one token is required")
        if capture_position is None:
            capture_position = len(tokens) - 1
        if capture_position < 0 or capture_position >= len(tokens):
            raise ValueError("capture_position is outside the supplied tokens")
        self.reset()
        logits: list[np.ndarray] = []
        captured: dict[str, dict[str, np.ndarray]] = {}
        for position, token in enumerate(tokens):
            output, modules = self.forward(
                int(token), position, capture=position == capture_position,
            )
            logits.append(output)
            if modules:
                captured = modules
        return tuple(logits), captured


def _tensor_metadata(value: np.ndarray) -> dict:
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def export_w4a8_module_goldens(
    checkpoint: Llama2CCheckpoint,
    output_directory: str | Path,
    tokens: Sequence[int],
    *,
    capture_position: int | None = None,
    tokenizer: Llama2CTokenizer | None = None,
) -> Path:
    """Write one independently hashed NPZ archive per module at one position."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    if capture_position is None:
        capture_position = len(tokens) - 1
    simulator = StoriesW4A8ModuleSimulator(checkpoint)
    all_logits, modules = simulator.run_tokens(tokens, capture_position=capture_position)
    records = []
    for name, tensors in modules.items():
        if not all(re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in name.split("/")):
            raise ValueError(f"unsafe module name: {name}")
        relative = Path(*name.split("/")).with_suffix(".npz")
        archive_path = output_directory / relative
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.ascontiguousarray(value) for key, value in tensors.items()}
        temporary = archive_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as output:
            np.savez_compressed(output, **arrays)
        temporary.replace(archive_path)
        records.append({
            "name": name,
            "archive": relative.as_posix(),
            "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "tensors": {key: _tensor_metadata(value) for key, value in arrays.items()},
        })

    selected_logits = np.asarray(all_logits[capture_position], dtype=np.float32)
    functional_results = StoriesFakeQuantReference(
        checkpoint, StoriesQuantizationSpec(4, 8, "per_output"),
    ).run_tokens(tokens, capture=False)
    functional_logits = np.asarray(functional_results[capture_position].logits, dtype=np.float32)
    top_indices = np.argsort(-selected_logits, kind="stable")[:16]
    manifest = {
        "format": "budgetcuda-stories-w4a8-module-goldens",
        "version": 1,
        "checkpoint": {
            "path": str(checkpoint.path),
            "sha256": checkpoint.sha256,
            "config": checkpoint.config.as_dict(),
        },
        "quantization": StoriesQuantizationSpec(4, 8, "per_output").as_dict(),
        "numeric_contract": {
            "weight_codes": "signed symmetric INT4 [-7,7]",
            "activation_codes": "signed symmetric INT8 [-127,127]",
            "probability_codes": "unsigned UINT8 [0,255]",
            "gemm_accumulator": "signed INT32",
            "weight_layout": "[N-tile][K][4 bytes], low nibble first",
            "activation_layout": "[M-tile][K][8 row lanes]",
        },
        "tokens": [int(token) for token in tokens],
        "decoded_tokens": tokenizer.decode(tokens) if tokenizer else None,
        "capture_position": int(capture_position),
        "input_token": int(tokens[capture_position]),
        "greedy_next_token": int(np.argmax(selected_logits)),
        "functional_fake_quant_comparison": {
            "reference_greedy_next_token": int(np.argmax(functional_logits)),
            "integer_greedy_next_token": int(np.argmax(selected_logits)),
            "top1_matches": bool(np.argmax(functional_logits) == np.argmax(selected_logits)),
            "max_abs_logit_error": float(np.max(np.abs(functional_logits - selected_logits))),
            "note": (
                "The functional reference multiplies dequantized FP32 operands; "
                "this module simulator uses explicit INT32 accumulation. Dynamic "
                "requantization can amplify reduction-rounding differences."
            ),
        },
        "top_logits": [
            {"token": int(index), "logit": float(selected_logits[index])}
            for index in top_indices
        ],
        "module_count": len(records),
        "modules": records,
    }
    manifest_path = output_directory / "manifest.json"
    temporary_manifest = output_directory / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return manifest_path


def validate_w4a8_module_goldens(manifest_path: str | Path) -> dict:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("format") != "budgetcuda-stories-w4a8-module-goldens"
            or manifest.get("version") != 1):
        raise ValueError("unsupported W4A8 module golden manifest")
    count = 0
    seen = set()
    for module in manifest.get("modules", []):
        relative = module.get("archive")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("module archive must be a safe relative path")
        if relative in seen:
            raise ValueError(f"duplicate module archive: {relative}")
        seen.add(relative)
        archive_path = manifest_path.parent / relative
        digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if digest != module.get("archive_sha256"):
            raise ValueError(f"module archive SHA-256 mismatch: {relative}")
        expected = module.get("tensors", {})
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise ValueError(f"module tensor keys differ: {relative}")
            for key, tensor in expected.items():
                value = np.ascontiguousarray(archive[key])
                if list(value.shape) != tensor.get("shape") or str(value.dtype) != tensor.get("dtype"):
                    raise ValueError(f"module tensor metadata mismatch: {relative}:{key}")
                if hashlib.sha256(value.tobytes()).hexdigest() != tensor.get("sha256"):
                    raise ValueError(f"module tensor SHA-256 mismatch: {relative}:{key}")
                count += 1
    if len(seen) != manifest.get("module_count"):
        raise ValueError("module_count does not match manifest entries")
    return {
        "manifest": str(manifest_path),
        "modules": len(seen),
        "tensors": count,
        "capture_position": manifest.get("capture_position"),
        "greedy_next_token": manifest.get("greedy_next_token"),
        "checkpoint_sha256": manifest.get("checkpoint", {}).get("sha256"),
    }
