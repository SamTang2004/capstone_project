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
    fc1_bias_int32=None,
    fc2_bias_int32=None,
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
    b1 = (0 if fc1_bias_int32 is None else
          graph.add_buffer("fc1.bias", (mlp_dim,), DType.INT32, BufferUsage.WEIGHT, fc1_bias_int32))
    q1_params = graph.requantize_params("fc1.requant", *fc1_requant)
    q1 = (graph.add_transformer_activation("fc1.q", tokens, mlp_dim) if accelerator_layout
          else graph.add_buffer("fc1.q", (tokens, mlp_dim), DType.INT8))
    gelu = (graph.add_transformer_activation("gelu", tokens, mlp_dim) if accelerator_layout
            else graph.add_buffer("gelu", (tokens, mlp_dim), DType.INT8))
    w2 = (graph.add_transformer_weight("fc2.weight", mlp_dim, hidden_dim,
                                       _matrix(fc2_weights, mlp_dim, hidden_dim), int4=weight_int4)
          if accelerator_layout else
          graph.add_buffer("fc2.weight", (mlp_dim, hidden_dim), DType.INT8, BufferUsage.WEIGHT, fc2_weights))
    b2 = (0 if fc2_bias_int32 is None else
          graph.add_buffer("fc2.bias", (hidden_dim,), DType.INT32, BufferUsage.WEIGHT, fc2_bias_int32))
    q2_params = graph.requantize_params("fc2.requant", *fc2_requant)
    q2 = graph.add_buffer("fc2.q", (tokens, hidden_dim), DType.INT8)
    out = graph.add_buffer("output", (tokens, hidden_dim), DType.INT8, BufferUsage.OUTPUT)
    graph.layer_norm(inp, gamma, beta, norm, rows=tokens, features=hidden_dim)
    if accelerator_layout:
        graph.transformer_gemm_requant(
            norm, w1, q1, q1_params, m=tokens, n=mlp_dim, k=hidden_dim,
            bias_idx=b1,
        )
    else:
        acc1 = graph.add_buffer("fc1.acc", (tokens, mlp_dim), DType.INT32)
        graph.gemm(norm, w1, acc1, m=tokens, n=mlp_dim, k=hidden_dim, bias_idx=b1)
        graph.requantize(acc1, q1, q1_params)
    graph.gelu(q1, gelu)
    if accelerator_layout:
        graph.transformer_gemm_requant(
            gelu, w2, q2, q2_params, m=tokens, n=hidden_dim, k=mlp_dim, bias_idx=b2)
    else:
        acc2 = graph.add_buffer("fc2.acc", (tokens, hidden_dim), DType.INT32)
        graph.gemm(gelu, w2, acc2, m=tokens, n=hidden_dim, k=mlp_dim, bias_idx=b2)
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
    q_bias_int32=None,
    k_bias_int32=None,
    v_bias_int32=None,
    output_bias_int32=None,
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
        for label, weights, biases in (("q", q_weights, q_bias_int32), ("k", k_weights, k_bias_int32),
                                       ("v", v_weights, v_bias_int32)):
            weight = graph.add_transformer_weight(
                f"{prefix}.{label}.weight", hidden_dim, head_dim,
                _head_columns(weights, hidden_dim, head_dim, head), int4=weight_int4,
            )
            bias = (0 if biases is None else graph.add_buffer(
                f"{prefix}.{label}.bias", (head_dim,), DType.INT32, BufferUsage.WEIGHT,
                list(biases)[head * head_dim:(head + 1) * head_dim]))
            if label == "k":
                quantized = graph.add_transformer_dynamic_b(f"{prefix}.key_b", head_dim, tokens)
            elif label == "v":
                quantized = graph.add_transformer_dynamic_b(f"{prefix}.value_b", tokens, head_dim)
            else:
                quantized = graph.add_transformer_activation(f"{prefix}.{label}", tokens, head_dim)
            graph.transformer_gemm_requant(
                norm, weight, quantized, qkv_params,
                m=tokens, n=head_dim, k=hidden_dim, bias_idx=bias,
                dynamic_b_transpose=label == "k",
            )
            projected.append(quantized)
        query, key_b, value_b = projected

        scores = graph.add_buffer(f"{prefix}.score", (tokens, tokens), DType.INT8)
        graph.transformer_gemm_requant(
            query, key_b, scores, score_params, m=tokens, n=tokens, k=head_dim)
        probabilities = graph.add_transformer_activation(
            f"{prefix}.probabilities_tiled", tokens, tokens)
        graph.softmax(scores, probabilities, rows=tokens, length=tokens)

        context = graph.add_buffer(f"{prefix}.context", (tokens, head_dim), DType.INT8)
        graph.transformer_gemm_requant(
            probabilities, value_b, context, context_params,
            m=tokens, n=head_dim, k=tokens)
        head_outputs.append(context)

    merged = graph.add_transformer_activation("heads.merged_tiled", tokens, hidden_dim)
    graph.token_concat(head_outputs, merged, rows=tokens, widths=[head_dim] * heads)
    out_weight = graph.add_transformer_weight(
        "output.weight", hidden_dim, hidden_dim,
        _matrix(output_weights, hidden_dim, hidden_dim), int4=weight_int4,
    )
    out_bias = (0 if output_bias_int32 is None else graph.add_buffer(
        "output.bias", (hidden_dim,), DType.INT32, BufferUsage.WEIGHT, output_bias_int32))
    out_params = graph.requantize_params("output.requant", *output_requant)
    projected_out = graph.add_buffer("output.q", (tokens, hidden_dim), DType.INT8)
    graph.transformer_gemm_requant(
        merged, out_weight, projected_out, out_params,
        m=tokens, n=hidden_dim, k=hidden_dim, bias_idx=out_bias)
    out = graph.add_buffer("output", (tokens, hidden_dim), DType.INT8, BufferUsage.OUTPUT)
    graph.residual_add(inp, projected_out, out)
    return graph.package(inp, out)


def build_deit_linear_package(
    rows: int,
    input_features: int,
    output_features: int,
    weights,
    *,
    bias_int32=None,
    requant=(1, 0, 0, 0),
    weight_int4: bool = False,
    name: str = "deit_linear",
) -> ModelPackage:
    """Build one tiled accelerator linear layer with row-major INT8 output."""
    graph = GraphBuilder(name)
    inp = graph.add_transformer_activation("input", rows, input_features)
    weight = graph.add_transformer_weight(
        "weight", input_features, output_features,
        _matrix(weights, input_features, output_features), int4=weight_int4,
    )
    bias = (0 if bias_int32 is None else graph.add_buffer(
        "bias", (output_features,), DType.INT32, BufferUsage.WEIGHT, bias_int32))
    params = graph.requantize_params("requant", *requant)
    output = graph.add_buffer("output", (rows, output_features), DType.INT8, BufferUsage.OUTPUT)
    graph.transformer_gemm_requant(
        inp, weight, output, params, m=rows, n=output_features,
        k=input_features, bias_idx=bias)
    return graph.package(inp, output)


def build_deit_layernorm_package(
    rows: int,
    features: int,
    gamma_q8,
    beta_q8,
    *,
    name: str = "deit_final_norm",
) -> ModelPackage:
    graph = GraphBuilder(name)
    inp = graph.add_buffer("input", (rows, features), DType.INT8)
    gamma = graph.add_buffer("gamma", (features,), DType.INT16, BufferUsage.WEIGHT, gamma_q8)
    beta = graph.add_buffer("beta", (features,), DType.INT16, BufferUsage.WEIGHT, beta_q8)
    output = graph.add_transformer_activation(
        "output", rows, features, usage=BufferUsage.OUTPUT)
    graph.layer_norm(inp, gamma, beta, output, rows=rows, features=features)
    return graph.package(inp, output)
