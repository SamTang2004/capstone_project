"""Short W4A8 QAT run for the llama2.c stories15M checkpoint.

The student uses per-output-channel symmetric W4 fake quantization and
operator-boundary dynamic A8 fake quantization. A frozen FP32 teacher provides
logit distillation while next-token cross entropy keeps the objective tied to
the official TinyStories training corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "FPGA"))

from budgetcuda.stories import Llama2CCheckpoint, Llama2CTokenizer  # noqa: E402


def _ste_symmetric(value: torch.Tensor, bits: int, dim: int | tuple[int, ...]) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    maximum = value.detach().abs().amax(dim=dim, keepdim=True)
    scale = torch.where(maximum > 0, maximum / qmax, torch.ones_like(maximum))
    dequantized = torch.round(value / scale).clamp(-qmax, qmax) * scale
    return value + (dequantized - value).detach()


def _ste_probability(value: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = (1 << bits) - 1
    dequantized = torch.round(value.clamp(0, 1) * qmax) / qmax
    return value + (dequantized - value).detach()


class StoriesQATModel(nn.Module):
    def __init__(self, checkpoint: Llama2CCheckpoint, *, quantized: bool):
        super().__init__()
        self.config = checkpoint.config
        self.quantized = quantized

        def parameter(value: np.ndarray) -> nn.Parameter:
            return nn.Parameter(torch.from_numpy(np.asarray(value, dtype=np.float32).copy()))

        self.token_embedding_table = parameter(checkpoint.token_embedding_table)
        self.rms_att_weight = parameter(checkpoint.rms_att_weight)
        self.wq = parameter(checkpoint.wq)
        self.wk = parameter(checkpoint.wk)
        self.wv = parameter(checkpoint.wv)
        self.wo = parameter(checkpoint.wo)
        self.rms_ffn_weight = parameter(checkpoint.rms_ffn_weight)
        self.w1 = parameter(checkpoint.w1)
        self.w2 = parameter(checkpoint.w2)
        self.w3 = parameter(checkpoint.w3)
        self.rms_final_weight = parameter(checkpoint.rms_final_weight)

    def _weight(self, value: torch.Tensor) -> torch.Tensor:
        return _ste_symmetric(value, 4, -1) if self.quantized else value

    def _activation(self, value: torch.Tensor) -> torch.Tensor:
        return _ste_symmetric(value, 8, -1) if self.quantized else value

    def _probability(self, value: torch.Tensor) -> torch.Tensor:
        return _ste_probability(value, 8) if self.quantized else value

    @staticmethod
    def _rmsnorm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return weight * value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1e-5)

    def _rope(self, value: torch.Tensor) -> torch.Tensor:
        # value: batch, time, heads, head_size
        _, time, _, head_size = value.shape
        positions = torch.arange(time, device=value.device, dtype=torch.float32)
        pair_index = torch.arange(0, head_size, 2, device=value.device, dtype=torch.float32)
        frequencies = torch.pow(torch.tensor(10000.0, device=value.device), -pair_index / head_size)
        angles = positions[:, None] * frequencies[None, :]
        cosine = angles.cos()[None, :, None, :]
        sine = angles.sin()[None, :, None, :]
        even, odd = value[..., 0::2], value[..., 1::2]
        return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        c = self.config
        if tokens.ndim != 2 or tokens.shape[1] > c.seq_len:
            raise ValueError("tokens must be batch x time within the checkpoint context")

        embedding = self._weight(self.token_embedding_table)
        x = self._activation(F.embedding(tokens, embedding))
        time = tokens.shape[1]
        causal = torch.ones((time, time), dtype=torch.bool, device=tokens.device).tril()

        q_weights = self._weight(self.wq)
        k_weights = self._weight(self.wk)
        v_weights = self._weight(self.wv)
        o_weights = self._weight(self.wo)
        w1_weights = self._weight(self.w1)
        w2_weights = self._weight(self.w2)
        w3_weights = self._weight(self.w3)

        for layer in range(c.n_layers):
            normalized = self._activation(self._rmsnorm(x, self.rms_att_weight[layer]))
            q = self._activation(F.linear(normalized, q_weights[layer]))
            k = self._activation(F.linear(normalized, k_weights[layer]))
            v = self._activation(F.linear(normalized, v_weights[layer]))
            q = self._rope(q.view(*q.shape[:2], c.n_heads, c.head_size))
            k = self._rope(k.view(*k.shape[:2], c.n_kv_heads, c.head_size))
            q = self._activation(q.reshape(*q.shape[:2], c.dim)).view(
                *q.shape[:2], c.n_heads, c.head_size
            )
            k = self._activation(k.reshape(*k.shape[:2], c.kv_dim)).view(
                *k.shape[:2], c.n_kv_heads, c.head_size
            )
            v = v.view(*v.shape[:2], c.n_kv_heads, c.head_size)

            if c.n_heads != c.n_kv_heads:
                multiplier = c.n_heads // c.n_kv_heads
                k = k.repeat_interleave(multiplier, dim=2)
                v = v.repeat_interleave(multiplier, dim=2)
            scores = torch.einsum("bthd,bshd->bhts", q, k) / math.sqrt(c.head_size)
            valid_scores = scores.masked_fill(~causal[None, None, :, :], 0.0)
            scores = self._activation(valid_scores).masked_fill(
                ~causal[None, None, :, :], float("-inf")
            )
            probabilities = self._probability(torch.softmax(scores, dim=-1))
            context = torch.einsum("bhts,bshd->bthd", probabilities, v)
            context = self._activation(context).reshape(tokens.shape[0], time, c.dim)
            context = self._activation(context)
            attention_output = self._activation(F.linear(context, o_weights[layer]))
            x = self._activation(x + attention_output)

            normalized = self._activation(self._rmsnorm(x, self.rms_ffn_weight[layer]))
            gate = self._activation(F.linear(normalized, w1_weights[layer]))
            up = self._activation(F.linear(normalized, w3_weights[layer]))
            silu_gate = self._activation(F.silu(gate))
            swiglu = self._activation(silu_gate * up)
            ffn_output = self._activation(F.linear(swiglu, w2_weights[layer]))
            x = self._activation(x + ffn_output)

        x = self._activation(self._rmsnorm(x, self.rms_final_weight))
        return F.linear(x, embedding)


def _training_sequences(
    path: Path, tokenizer: Llama2CTokenizer, *, max_stories: int, sequence_length: int,
) -> torch.Tensor:
    text = path.read_text(encoding="utf-8", errors="ignore")
    stories = [item.strip() for item in text.split("<|endoftext|>") if item.strip()]
    stream: list[int] = []
    for story in stories[:max_stories]:
        stream.extend(tokenizer.encode(story, bos=True, eos=True))
    width = sequence_length + 1
    count = len(stream) // width
    if count < 2:
        raise ValueError("training corpus does not contain two complete sequences")
    return torch.tensor(stream[:count * width], dtype=torch.long).view(count, width)


def _write_checkpoint(model: StoriesQATModel, source: Llama2CCheckpoint, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    c = source.config
    signed_vocab = c.vocab_size if c.shared_classifier else -c.vocab_size
    names = (
        "token_embedding_table", "rms_att_weight", "wq", "wk", "wv", "wo",
        "rms_ffn_weight", "w1", "w2", "w3", "rms_final_weight",
    )
    with path.open("wb") as output:
        output.write(struct.pack(
            "<7i", c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads,
            signed_vocab, c.seq_len,
        ))
        for name in names:
            value = getattr(model, name).detach().cpu().numpy().astype("<f4", copy=False)
            output.write(value.tobytes(order="C"))
        output.write(np.asarray(source.legacy_rope_real, dtype="<f4").tobytes(order="C"))
        output.write(np.asarray(source.legacy_rope_imag, dtype="<f4").tobytes(order="C"))
        if not c.shared_classifier:
            raise NotImplementedError("QAT export currently expects the shared stories15M classifier")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("train_corpus", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--max-stories", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--distillation-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch size must be positive")
    if not 0 <= args.distillation_weight <= 1:
        raise ValueError("distillation weight must be in [0,1]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    source = Llama2CCheckpoint(args.checkpoint, require_stories15m=True)
    tokenizer = Llama2CTokenizer(args.tokenizer, source.config.vocab_size)
    sequences = _training_sequences(
        args.train_corpus, tokenizer, max_stories=args.max_stories,
        sequence_length=args.sequence_length,
    )
    device = torch.device(args.device)
    student = StoriesQATModel(source, quantized=True).to(device)
    teacher = StoriesQATModel(source, quantized=False).to(device).eval()
    teacher.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(sequences), generator=generator)
    cursor = 0
    losses: list[dict] = []
    started = perf_counter()

    student.train()
    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > len(order):
            order = torch.randperm(len(sequences), generator=generator)
            cursor = 0
        batch = sequences[order[cursor:cursor + args.batch_size]].to(device)
        cursor += args.batch_size
        inputs, targets = batch[:, :-1], batch[:, 1:]

        with torch.no_grad():
            teacher_logits = teacher(inputs)
        student_logits = student(inputs)
        cross_entropy = F.cross_entropy(
            student_logits.reshape(-1, source.config.vocab_size), targets.reshape(-1)
        )
        temperature = args.temperature
        distillation = F.kl_div(
            F.log_softmax(student_logits / temperature, dim=-1),
            F.softmax(teacher_logits / temperature, dim=-1),
            reduction="batchmean",
        ) * (temperature * temperature) / inputs.shape[1]
        loss = ((1 - args.distillation_weight) * cross_entropy
                + args.distillation_weight * distillation)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        item = {
            "step": step,
            "loss": float(loss.detach()),
            "cross_entropy": float(cross_entropy.detach()),
            "distillation_kl": float(distillation.detach()),
            "gradient_norm": float(gradient_norm),
            "learning_rate": float(scheduler.get_last_lr()[0]),
        }
        losses.append(item)
        if step == 1 or step % 10 == 0 or step == args.steps:
            print(json.dumps(item), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "stories15M_w4a8_qat_float.bin"
    _write_checkpoint(student, source, checkpoint_path)
    exported = Llama2CCheckpoint(checkpoint_path, require_stories15m=True)
    report = {
        "format": "budgetcuda-stories-w4a8-qat-training",
        "version": 1,
        "source_checkpoint_sha256": source.sha256,
        "exported_checkpoint_sha256": exported.sha256,
        "train_corpus": str(args.train_corpus),
        "train_sequences": len(sequences),
        "train_prediction_tokens_per_step": args.batch_size * args.sequence_length,
        "quantization": {
            "weights": "symmetric W4 per-output-channel STE fake quant",
            "activations": "dynamic symmetric A8 per-vector STE fake quant",
            "probabilities": "unsigned A8 STE fake quant",
            "accumulation": "FP32",
            "rmsnorm_parameters": "FP32",
            "embedding_classifier_tied": True,
        },
        "hyperparameters": vars(args) | {
            "checkpoint": str(args.checkpoint), "tokenizer": str(args.tokenizer),
            "train_corpus": str(args.train_corpus), "output_dir": str(args.output_dir),
        },
        "elapsed_seconds": perf_counter() - started,
        "history": losses,
    }
    report_path = args.output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "checkpoint": str(checkpoint_path), "report": str(report_path),
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
