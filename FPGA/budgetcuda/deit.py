from __future__ import annotations

from dataclasses import dataclass

from .compiler import GraphBuilder, ModelPackage
from .dtypes import BufferUsage, DType


def _matrix(values, rows: int, columns: int):
    values = list(values)
    if len(values) == rows and all(hasattr(row, "__len__") for row in values):
        matrix = [list(row) for row in values]
        if any(len(row) != columns for row in matrix):
            raise ValueError(f"Expected {rows}x{columns} weight values")
        return matrix
    if len(values) != rows * columns:
        raise ValueError(f"Expected {rows}x{columns} weight values")
    return [values[row * columns : (row + 1) * columns] for row in range(rows)]


def _head_columns(values, hidden_dim: int, head_dim: int, head: int):
    matrix = _matrix(values, hidden_dim, hidden_dim)
    start = head * head_dim
    return [row[start:start + head_dim] for row in matrix]


@dataclass(frozen=True)
class DeiTConfig:
    image_size: int = 224
    patch_size: int = 16
    hidden_dim: int = 192
    heads: int = 3
    depth: int = 12
    mlp_dim: int = 768
    classes: int = 6

    def __post_init__(self):
        if self.image_size % self.patch_size or self.hidden_dim % self.heads:
            raise ValueError("DeiT image/patch and hidden/head dimensions must divide evenly")

    @property
    def tokens(self) -> int:
        return (self.image_size // self.patch_size) ** 2 + 1


def build_deit_mlp_block_package(
    tokens: int,
    hidden_dim: int,
    mlp_dim: int,
    fc1_weights,
    fc2_weights,
    *,
    gamma_q8=None,
    beta_q8=None,
    fc1_requant=(1, 0, 0, 0),
    fc2_requant=(1, 0, 0, 0),
    accelerator_layout: bool = False,
    weight_int4: bool = False,
    name: str = "deit_mlp",
) -> ModelPackage:
    """Build a pre-norm DeiT MLP residual block using the shared graph ABI."""
    if min(tokens, hidden_dim, mlp_dim) <= 0:
        raise ValueError("DeiT MLP dimensions must be positive")
    gamma_q8 = [256] * hidden_dim if gamma_q8 is None else gamma_q8
    beta_q8 = [0] * hidden_dim if beta_q8 is None else beta_q8

    graph = GraphBuilder(name)
    inp = graph.add_buffer("input", (tokens, hidden_dim), DType.INT8)
    gamma = graph.add_buffer("norm.gamma", (hidden_dim,), DType.INT16, BufferUsage.WEIGHT, gamma_q8)
    beta = graph.add_buffer("norm.beta", (hidden_dim,), DType.INT16, BufferUsage.WEIGHT, beta_q8)
    norm = (graph.add_transformer_activation("norm.output", tokens, hidden_dim) if accelerator_layout
            else graph.add_buffer("norm.output", (tokens, hidden_dim), DType.INT8))
    w1 = (graph.add_transformer_weight("fc1.weight", hidden_dim, mlp_dim,
                                       _matrix(fc1_weights, hidden_dim, mlp_dim), int4=weight_int4)
          if accelerator_layout else
          graph.add_buffer("fc1.weight", (hidden_dim, mlp_dim), DType.INT8, BufferUsage.WEIGHT, fc1_weights))
    acc1 = graph.add_buffer("fc1.acc", (tokens, mlp_dim), DType.INT32)
    q1_params = graph.requantize_params("fc1.requant", *fc1_requant)
    q1 = (graph.add_transformer_activation("fc1.q", tokens, mlp_dim) if accelerator_layout
          else graph.add_buffer("fc1.q", (tokens, mlp_dim), DType.INT8))
    gelu = (graph.add_transformer_activation("gelu", tokens, mlp_dim) if accelerator_layout
            else graph.add_buffer("gelu", (tokens, mlp_dim), DType.INT8))
    w2 = (graph.add_transformer_weight("fc2.weight", mlp_dim, hidden_dim,
                                       _matrix(fc2_weights, mlp_dim, hidden_dim), int4=weight_int4)
          if accelerator_layout else
          graph.add_buffer("fc2.weight", (mlp_dim, hidden_dim), DType.INT8, BufferUsage.WEIGHT, fc2_weights))
    acc2 = graph.add_buffer("fc2.acc", (tokens, hidden_dim), DType.INT32)
    q2_params = graph.requantize_params("fc2.requant", *fc2_requant)
    q2 = graph.add_buffer("fc2.q", (tokens, hidden_dim), DType.INT8)
    out = graph.add_buffer("output", (tokens, hidden_dim), DType.INT8, BufferUsage.OUTPUT)
    graph.layer_norm(inp, gamma, beta, norm, rows=tokens, features=hidden_dim)
    (graph.transformer_gemm if accelerator_layout else graph.gemm)(norm, w1, acc1, m=tokens, n=mlp_dim, k=hidden_dim)
    if accelerator_layout:
        graph.requantize_pack(acc1, q1, q1_params, rows=tokens, columns=mlp_dim)
    else:
        graph.requantize(acc1, q1, q1_params)
    graph.gelu(q1, gelu)
    (graph.transformer_gemm if accelerator_layout else graph.gemm)(gelu, w2, acc2, m=tokens, n=hidden_dim, k=mlp_dim)
    graph.requantize(acc2, q2, q2_params)
    graph.residual_add(inp, q2, out)
    return graph.package(inp, out)


def build_deit_attention_block_package(
    tokens: int,
    hidden_dim: int,
    heads: int,
    q_weights,
    k_weights,
    v_weights,
    output_weights,
    *,
    gamma_q8=None,
    beta_q8=None,
    qkv_requant=(1, 0, 0, 0),
    score_requant=(1, 0, 0, 0),
    context_requant=(1, 0, 0, 0),
    output_requant=(1, 0, 0, 0),
    weight_int4: bool = False,
    name: str = "deit_attention",
) -> ModelPackage:
    """Build a complete pre-norm DeiT multi-head self-attention residual block."""
    if tokens <= 0 or hidden_dim <= 0 or not 1 <= heads <= 3 or hidden_dim % heads:
        raise ValueError("Attention requires positive dimensions and one to three equal heads")
    head_dim = hidden_dim // heads
    gamma_q8 = [256] * hidden_dim if gamma_q8 is None else gamma_q8
    beta_q8 = [0] * hidden_dim if beta_q8 is None else beta_q8
    graph = GraphBuilder(name)
    inp = graph.add_buffer("input", (tokens, hidden_dim), DType.INT8)
    gamma = graph.add_buffer("norm.gamma", (hidden_dim,), DType.INT16, BufferUsage.WEIGHT, gamma_q8)
    beta = graph.add_buffer("norm.beta", (hidden_dim,), DType.INT16, BufferUsage.WEIGHT, beta_q8)
    norm = graph.add_transformer_activation("norm.output", tokens, hidden_dim)
    graph.layer_norm(inp, gamma, beta, norm, rows=tokens, features=hidden_dim)
    qkv_params = graph.requantize_params("qkv.requant", *qkv_requant)
    score_params = graph.requantize_params("score.requant", *score_requant)
    context_params = graph.requantize_params("context.requant", *context_requant)
    head_outputs = []

    for head in range(heads):
        prefix = f"head{head}"
        projected = []
        for label, weights in (("q", q_weights), ("k", k_weights), ("v", v_weights)):
            weight = graph.add_transformer_weight(
                f"{prefix}.{label}.weight", hidden_dim, head_dim,
                _head_columns(weights, hidden_dim, head_dim, head), int4=weight_int4,
            )
            acc = graph.add_buffer(f"{prefix}.{label}.acc", (tokens, head_dim), DType.INT32)
            quantized = graph.add_transformer_activation(f"{prefix}.{label}", tokens, head_dim)
            graph.transformer_gemm(norm, weight, acc, m=tokens, n=head_dim, k=hidden_dim)
            graph.requantize_pack(acc, quantized, qkv_params, rows=tokens, columns=head_dim)
            projected.append(quantized)
        query, key, value = projected

        key_b = graph.add_transformer_dynamic_b(f"{prefix}.key_b", head_dim, tokens)
        graph.activation_b_pack(key, key_b, reduction=head_dim, outputs=tokens, transpose=True)
        score_acc = graph.add_buffer(f"{prefix}.score.acc", (tokens, tokens), DType.INT32)
        graph.transformer_gemm(query, key_b, score_acc, m=tokens, n=tokens, k=head_dim)
        scores = graph.add_buffer(f"{prefix}.score", (tokens, tokens), DType.INT8)
        graph.requantize(score_acc, scores, score_params)
        probabilities = graph.add_buffer(f"{prefix}.probabilities", (tokens, tokens), DType.INT8)
        graph.softmax(scores, probabilities, rows=tokens, length=tokens)
        probabilities_tiled = graph.add_transformer_activation(f"{prefix}.probabilities_tiled", tokens, tokens)
        graph.activation_a_pack(probabilities, probabilities_tiled, rows=tokens, features=tokens)

        value_b = graph.add_transformer_dynamic_b(f"{prefix}.value_b", tokens, head_dim)
        graph.activation_b_pack(value, value_b, reduction=tokens, outputs=head_dim)
        context_acc = graph.add_buffer(f"{prefix}.context.acc", (tokens, head_dim), DType.INT32)
        graph.transformer_gemm(probabilities_tiled, value_b, context_acc, m=tokens, n=head_dim, k=tokens)
        context = graph.add_buffer(f"{prefix}.context", (tokens, head_dim), DType.INT8)
        graph.requantize(context_acc, context, context_params)
        head_outputs.append(context)

    merged = graph.add_buffer("heads.merged", (tokens, hidden_dim), DType.INT8)
    graph.token_concat(head_outputs, merged, rows=tokens, widths=[head_dim] * heads)
    merged_tiled = graph.add_transformer_activation("heads.merged_tiled", tokens, hidden_dim)
    graph.activation_a_pack(merged, merged_tiled, rows=tokens, features=hidden_dim)
    out_weight = graph.add_transformer_weight(
        "output.weight", hidden_dim, hidden_dim,
        _matrix(output_weights, hidden_dim, hidden_dim), int4=weight_int4,
    )
    out_acc = graph.add_buffer("output.acc", (tokens, hidden_dim), DType.INT32)
    graph.transformer_gemm(merged_tiled, out_weight, out_acc, m=tokens, n=hidden_dim, k=hidden_dim)
    out_params = graph.requantize_params("output.requant", *output_requant)
    projected_out = graph.add_buffer("output.q", (tokens, hidden_dim), DType.INT8)
    graph.requantize(out_acc, projected_out, out_params)
    out = graph.add_buffer("output", (tokens, hidden_dim), DType.INT8, BufferUsage.OUTPUT)
    graph.residual_add(inp, projected_out, out)
    return graph.package(inp, out)
