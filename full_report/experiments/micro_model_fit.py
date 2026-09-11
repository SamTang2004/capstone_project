"""Reproducible static-fit probes for small Transformer model shapes.

This deliberately does not claim model accuracy or board throughput.  It asks
whether representative dense Transformer blocks can be expressed by the
current BudgetCUDA graph builder and whether their persistent tensors fit a
64 MiB model-size budget.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FPGA"))

from budgetcuda.compiler import compose_model_packages  # noqa: E402
from budgetcuda.deit import (  # noqa: E402
    build_deit_attention_block_package,
    build_deit_mlp_block_package,
)
from budgetcuda.dtypes import BufferUsage  # noqa: E402


MIB = 1024 * 1024
HARDWARE_MAX_K = 1024
HARDWARE_MAX_SOFTMAX = 256
BUILDER_MAX_HEADS = 3


@dataclass(frozen=True)
class Candidate:
    name: str
    family: str
    layers: int
    tokens: int
    hidden: int
    heads: int
    mlp: int
    vocab: int
    causal: bool = False


CANDIDATES = (
    Candidate("bert_tiny_shaped", "encoder", 2, 128, 128, 2, 512, 30_522),
    Candidate("byte_encoder_12l", "encoder", 12, 256, 192, 3, 768, 256),
    Candidate("micro_gpt_12l", "decoder", 12, 256, 192, 3, 768, 8_192, causal=True),
    Candidate("wide_encoder_6l", "encoder", 6, 256, 384, 3, 1024, 8_192),
    Candidate("near_budget_decoder_12l", "decoder", 12, 256, 768, 3, 1024, 8_192, causal=True),
)


def parameter_counts(model: Candidate) -> dict[str, int]:
    # Four attention projections and two MLP matrices. Biases and two affine
    # norms are counted separately at their natural deployment widths.
    matrix_per_layer = 4 * model.hidden * model.hidden + 2 * model.hidden * model.mlp
    bias_per_layer = 4 * model.hidden + model.mlp + model.hidden
    norm_per_layer = 4 * model.hidden
    embedding = model.vocab * model.hidden
    return {
        "matrix_weights": matrix_per_layer * model.layers,
        "bias_values": bias_per_layer * model.layers,
        "norm_values": norm_per_layer * model.layers,
        "token_embedding_values": embedding,
        "lm_head_values": embedding if model.family == "decoder" else 0,
    }


def storage_bytes(model: Candidate, weight_bits: int, tied_lm_head: bool = True) -> int:
    counts = parameter_counts(model)
    matrices = (counts["matrix_weights"] * weight_bits + 7) // 8
    # The current datapath uses INT32 bias and INT16 norm coefficients.
    auxiliaries = counts["bias_values"] * 4 + counts["norm_values"] * 2
    embeddings = counts["token_embedding_values"]  # proposed INT8 lookup table
    lm_head = 0
    if counts["lm_head_values"] and not tied_lm_head:
        lm_head = (counts["lm_head_values"] * weight_bits + 7) // 8
    return matrices + auxiliaries + embeddings + lm_head


def macs_per_full_sequence(model: Candidate) -> int:
    # Dense self-attention: Q/K/V/O projections, QK^T and AV, plus the MLP.
    t, d, f = model.tokens, model.hidden, model.mlp
    per_layer = 4 * t * d * d + 2 * t * t * d + 2 * t * d * f
    return model.layers * per_layer


def compile_one_block(model: Candidate, weight_bits: int) -> dict[str, object]:
    if model.heads > BUILDER_MAX_HEADS:
        return {"compiled": False, "reason": "graph builder supports at most three heads"}
    if max(model.hidden, model.mlp, model.tokens) > HARDWARE_MAX_K:
        return {"compiled": False, "reason": "a GEMM reduction exceeds MAX_K=1024"}
    if model.tokens > HARDWARE_MAX_SOFTMAX:
        return {"compiled": False, "reason": "attention Softmax exceeds 256 elements"}

    d, f = model.hidden, model.mlp
    zeros_d2 = [0] * (d * d)
    attention = build_deit_attention_block_package(
        model.tokens,
        d,
        model.heads,
        zeros_d2,
        zeros_d2,
        zeros_d2,
        zeros_d2,
        weight_int4=weight_bits == 4,
        name=f"{model.name}.attention",
    )
    mlp = build_deit_mlp_block_package(
        model.tokens,
        d,
        f,
        [0] * (d * f),
        [0] * (f * d),
        accelerator_layout=True,
        weight_int4=weight_bits == 4,
        name=f"{model.name}.mlp",
    )
    block = compose_model_packages(f"{model.name}.block", (attention, mlp))
    constant_bytes = sum(
        buffer.size_bytes for buffer in block.buffers
        if buffer.usage == BufferUsage.WEIGHT
    )
    activation_bytes = sum(
        buffer.size_bytes for buffer in block.buffers
        if buffer.usage != BufferUsage.WEIGHT
    )
    return {
        "compiled": True,
        "layers_in_graph": len(block.layers),
        "buffers_in_graph": len(block.buffers),
        "descriptor_bytes": len(block.serialize_descriptor()),
        "one_block_constant_bytes": constant_bytes,
        "one_block_activation_bytes_without_alias_planning": activation_bytes,
        "semantic_gaps": (
            ["causal attention mask", "KV-cache update", "token embedding lookup", "logit sampling"]
            if model.causal else
            ["token embedding lookup", "attention/padding mask"]
        ),
    }


def main() -> None:
    rows = []
    for model in CANDIDATES:
        counts = parameter_counts(model)
        rows.append({
            **asdict(model),
            **counts,
            "dense_sequence_macs": macs_per_full_sequence(model),
            "estimated_w8_bytes": storage_bytes(model, 8),
            "estimated_w4_bytes": storage_bytes(model, 4),
            "under_64mib_w8": storage_bytes(model, 8) < 64 * MIB,
            "under_64mib_w4": storage_bytes(model, 4) < 64 * MIB,
            "w8_probe": compile_one_block(model, 8),
            "w4_probe": compile_one_block(model, 4),
        })

    output = {
        "method": "Analytic tensor sizing plus construction of one attention+MLP block with the repository graph builder",
        "limits_used": {
            "max_gemm_reduction": HARDWARE_MAX_K,
            "max_attention_softmax_length": HARDWARE_MAX_SOFTMAX,
            "max_builder_heads": BUILDER_MAX_HEADS,
            "model_budget_bytes": 64 * MIB,
        },
        "candidates": rows,
    }
    result_path = Path(__file__).with_name("micro_model_fit_results.json")
    result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
