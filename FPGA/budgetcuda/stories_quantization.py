from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from .stories import (Llama2CCheckpoint, Llama2CTokenizer,
                      StoriesFloatReference, export_stories_golden)


@dataclass(frozen=True)
class StoriesQuantizationSpec:
    weight_bits: int
    activation_bits: int
    weight_granularity: str = "per_tensor"

    def __post_init__(self) -> None:
        if not 2 <= int(self.weight_bits) <= 16:
            raise ValueError("weight_bits must be in 2..16")
        if not 2 <= int(self.activation_bits) <= 16:
            raise ValueError("activation_bits must be in 2..16")
        if self.weight_granularity not in ("per_tensor", "per_output"):
            raise ValueError("weight_granularity must be per_tensor or per_output")

    @property
    def label(self) -> str:
        return f"W{self.weight_bits}A{self.activation_bits}"

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "weight_bits": self.weight_bits,
            "activation_bits": self.activation_bits,
            "weight_granularity": self.weight_granularity,
        }


def _symmetric_fake_quant(value: np.ndarray, bits: int, *, axes=None) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    qmax = np.float32((1 << (bits - 1)) - 1)
    maximum = np.max(np.abs(value), axis=axes, keepdims=axes is not None)
    scale = np.asarray(maximum / qmax, dtype=np.float32)
    scale = np.where(scale == 0, np.float32(1.0), scale).astype(np.float32)
    quantized = np.clip(np.rint(value / scale), -qmax, qmax)
    return np.asarray(quantized * scale, dtype=np.float32)


def _unsigned_probability_fake_quant(value: np.ndarray, bits: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    qmax = np.float32((1 << bits) - 1)
    return np.asarray(np.rint(np.clip(value, 0.0, 1.0) * qmax) / qmax, dtype=np.float32)


def _quantize_weight(value: np.ndarray, spec: StoriesQuantizationSpec) -> np.ndarray:
    if spec.weight_granularity == "per_output":
        axes = -1
    elif value.ndim >= 3:
        axes = tuple(range(1, value.ndim))
    else:
        axes = None
    return _symmetric_fake_quant(value, spec.weight_bits, axes=axes)


class _FakeQuantWeights:
    def __init__(self, checkpoint: Llama2CCheckpoint, spec: StoriesQuantizationSpec):
        self.config = checkpoint.config
        self.token_embedding_table = _quantize_weight(checkpoint.token_embedding_table, spec)
        self.rms_att_weight = checkpoint.rms_att_weight
        self.wq = _quantize_weight(checkpoint.wq, spec)
        self.wk = _quantize_weight(checkpoint.wk, spec)
        self.wv = _quantize_weight(checkpoint.wv, spec)
        self.wo = _quantize_weight(checkpoint.wo, spec)
        self.rms_ffn_weight = checkpoint.rms_ffn_weight
        self.w1 = _quantize_weight(checkpoint.w1, spec)
        self.w2 = _quantize_weight(checkpoint.w2, spec)
        self.w3 = _quantize_weight(checkpoint.w3, spec)
        self.rms_final_weight = checkpoint.rms_final_weight
        self.wcls = (
            self.token_embedding_table
            if checkpoint.config.shared_classifier
            else _quantize_weight(checkpoint.wcls, spec)
        )


class StoriesFakeQuantReference(StoriesFloatReference):
    """Operator-boundary dynamic fake quantization for WXAY sensitivity sweeps."""

    def __init__(self, checkpoint: Llama2CCheckpoint, spec: StoriesQuantizationSpec):
        self.quantization = spec
        super().__init__(checkpoint)
        self.weights = _FakeQuantWeights(checkpoint, spec)

    def _activation(self, value: np.ndarray, _name: str) -> np.ndarray:
        return _symmetric_fake_quant(value, self.quantization.activation_bits)

    def _probability(self, value: np.ndarray, _name: str) -> np.ndarray:
        return _unsigned_probability_fake_quant(value, self.quantization.activation_bits)


def export_quantized_stories_golden(
    checkpoint: Llama2CCheckpoint,
    output_prefix: str | Path,
    prompt_tokens: Sequence[int],
    spec: StoriesQuantizationSpec,
    *,
    max_new_tokens: int = 0,
    top_k: int = 16,
    tokenizer: Llama2CTokenizer | None = None,
    provenance: dict | None = None,
) -> tuple[Path, Path]:
    """Export the actual WXAY fake-quant trace used by the host reference."""

    policy = {
        "spec": spec.as_dict(),
        "weights": "symmetric fake quant",
        "activations": "dynamic symmetric fake quant at operator boundaries",
        "attention_probabilities": "dynamic unsigned fake quant in [0,1]",
        "accumulation": "FP32, modeling a sufficiently wide INT32 accumulator",
        "rmsnorm_parameters": "FP32",
    }
    extra: dict[str, object] = {"quantization": policy}
    if provenance:
        extra["provenance"] = provenance
    return export_stories_golden(
        checkpoint, output_prefix, prompt_tokens,
        max_new_tokens=max_new_tokens, top_k=top_k, tokenizer=tokenizer,
        reference=StoriesFakeQuantReference(checkpoint, spec),
        golden_format="budgetcuda-stories-wxay-golden",
        extra_metadata=extra,
    )


def _log_softmax(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    maximum = np.max(value)
    shifted = value - maximum
    return shifted - math.log(float(np.sum(np.exp(shifted))))


def _grade(metrics: dict) -> tuple[str, list[str]]:
    reasons: list[str] = []
    finite = all(math.isfinite(float(metrics[name])) for name in (
        "quantized_perplexity", "perplexity_ratio", "top1_agreement",
        "mean_logit_cosine", "mean_normalized_rmse", "mean_kl_nats",
    ))
    if not finite:
        return "F", ["non-finite metric"]
    if metrics["perplexity_ratio"] >= 2.0:
        reasons.append("perplexity at least doubled")
    if metrics["top1_agreement"] < 0.50:
        reasons.append("FP32 top-1 agreement below 50%")
    if metrics["mean_logit_cosine"] < 0.90:
        reasons.append("mean logit cosine below 0.90")
    if reasons:
        return "F", reasons
    if (metrics["perplexity_ratio"] >= 1.50 or metrics["top1_agreement"] < 0.65 or
            metrics["mean_logit_cosine"] < 0.95):
        return "C", ["severe but not fatal degradation"]
    if (metrics["perplexity_ratio"] >= 1.15 or metrics["top1_agreement"] < 0.85 or
            metrics["mean_logit_cosine"] < 0.98):
        return "B", ["material degradation"]
    if (metrics["perplexity_ratio"] >= 1.05 or metrics["top1_agreement"] < 0.95 or
            metrics["mean_logit_cosine"] < 0.995):
        return "A", ["mild degradation"]
    return "S", ["within strict FP32 similarity thresholds"]


def _baseline_logits(
    checkpoint: Llama2CCheckpoint, sequences: Sequence[Sequence[int]],
) -> tuple[list[tuple[np.ndarray, ...]], list[tuple[int, ...]], float, int]:
    reference = StoriesFloatReference(checkpoint)
    all_logits: list[tuple[np.ndarray, ...]] = []
    all_targets: list[tuple[int, ...]] = []
    nll = 0.0
    count = 0
    for sequence in sequences:
        if len(sequence) < 2:
            continue
        results = reference.run_tokens(sequence[:-1], capture=False)
        logits = tuple(np.asarray(result.logits, dtype=np.float32) for result in results)
        targets = tuple(int(token) for token in sequence[1:])
        for value, target in zip(logits, targets):
            nll -= float(_log_softmax(value)[target])
            count += 1
        all_logits.append(logits)
        all_targets.append(targets)
    if not count:
        raise ValueError("evaluation requires at least one next-token target")
    return all_logits, all_targets, nll / count, count


def evaluate_quantization_sweep(
    checkpoint: Llama2CCheckpoint,
    sequences: Sequence[Sequence[int]],
    specs: Sequence[StoriesQuantizationSpec],
    *,
    baseline_checkpoint: Llama2CCheckpoint | None = None,
    sample_prompt_tokens: Sequence[int] | None = None,
    sample_new_tokens: int = 0,
) -> dict:
    """Compare WXAY configurations against one cached FP32 teacher run.

    ``checkpoint`` supplies the candidate weights. ``baseline_checkpoint`` can
    retain the original pretrained FP32 model when evaluating a fine-tuned or
    QAT checkpoint; by default the candidate's own float weights are used.
    """

    started = perf_counter()
    teacher = baseline_checkpoint or checkpoint
    if teacher.config != checkpoint.config:
        raise ValueError("baseline and candidate checkpoint configs must match")
    baseline, targets, baseline_nll, prediction_count = _baseline_logits(teacher, sequences)
    baseline_perplexity = math.exp(baseline_nll)
    candidate_float_perplexity = baseline_perplexity
    if teacher is not checkpoint:
        _, _, candidate_nll, candidate_count = _baseline_logits(checkpoint, sequences)
        if candidate_count != prediction_count:
            raise AssertionError("candidate and baseline prediction counts differ")
        candidate_float_perplexity = math.exp(candidate_nll)
    baseline_sample: tuple[int, ...] = ()
    if sample_prompt_tokens and sample_new_tokens:
        baseline_sample, _ = StoriesFloatReference(teacher).generate_greedy(
            sample_prompt_tokens, sample_new_tokens, capture=False,
        )

    evaluations: list[dict] = []
    for spec in specs:
        config_started = perf_counter()
        reference = StoriesFakeQuantReference(checkpoint, spec)
        quant_nll = 0.0
        top1_matches = 0
        cosine_sum = 0.0
        nrmse_sum = 0.0
        kl_sum = 0.0
        first_divergence = None
        global_position = 0
        for sequence, fp_logits, sequence_targets in zip(sequences, baseline, targets):
            if len(sequence) < 2:
                continue
            quant_results = reference.run_tokens(sequence[:-1], capture=False)
            for fp_value, quant_result, target in zip(fp_logits, quant_results, sequence_targets):
                quant_value = np.asarray(quant_result.logits, dtype=np.float32)
                fp_logp = _log_softmax(fp_value)
                quant_logp = _log_softmax(quant_value)
                quant_nll -= float(quant_logp[target])
                fp_top1 = int(np.argmax(fp_value))
                quant_top1 = int(np.argmax(quant_value))
                if fp_top1 == quant_top1:
                    top1_matches += 1
                elif first_divergence is None:
                    first_divergence = global_position
                fp64 = np.asarray(fp_value, dtype=np.float64)
                quant64 = np.asarray(quant_value, dtype=np.float64)
                denominator = float(np.linalg.norm(fp64) * np.linalg.norm(quant64))
                if denominator:
                    cosine_sum += float(np.dot(fp64, quant64) / denominator)
                else:
                    cosine_sum += 1.0 if not np.any(fp64) and not np.any(quant64) else 0.0
                fp_norm = float(np.linalg.norm(fp64))
                nrmse_sum += float(np.linalg.norm(quant64 - fp64) / fp_norm) if fp_norm else 0.0
                probability = np.exp(fp_logp)
                kl_sum += float(np.sum(probability * (fp_logp - quant_logp)))
                global_position += 1

        average_nll = quant_nll / prediction_count
        quantized_perplexity = math.exp(average_nll) if average_nll < 700 else math.inf
        metrics = {
            **spec.as_dict(),
            "prediction_tokens": prediction_count,
            "baseline_perplexity": baseline_perplexity,
            "quantized_perplexity": quantized_perplexity,
            "perplexity_ratio": quantized_perplexity / baseline_perplexity,
            "top1_agreement": top1_matches / prediction_count,
            "mean_logit_cosine": cosine_sum / prediction_count,
            "mean_normalized_rmse": nrmse_sum / prediction_count,
            "mean_kl_nats": kl_sum / prediction_count,
            "first_teacher_forced_top1_divergence": first_divergence,
            "elapsed_seconds": perf_counter() - config_started,
        }
        if baseline_sample:
            quant_sample, _ = reference.generate_greedy(
                sample_prompt_tokens, sample_new_tokens, capture=False,
            )
            prefix = 0
            for fp_token, quant_token in zip(baseline_sample, quant_sample):
                if fp_token != quant_token:
                    break
                prefix += 1
            metrics["greedy_tokens"] = list(quant_sample)
            metrics["greedy_prefix_match_tokens"] = prefix
        grade, reasons = _grade(metrics)
        metrics["grade"] = grade
        metrics["grade_reasons"] = reasons
        evaluations.append(metrics)

    return {
        "format": "budgetcuda-stories-wxay-sweep",
        "version": 1,
        "policy": {
            "weights": "symmetric fake quant; scale granularity is recorded in each evaluation",
            "activations": "dynamic symmetric fake quant at operator boundaries",
            "attention_probabilities": "dynamic unsigned fake quant in [0,1]",
            "accumulation": "FP32, modeling a sufficiently wide INT32 accumulator",
            "rmsnorm_parameters": "FP32",
            "grading": {
                "S": "ppl ratio <1.05, top1 >=0.95, cosine >=0.995",
                "A": "mild degradation",
                "B": "material degradation",
                "C": "severe degradation",
                "F": "ppl ratio >=2, top1 <0.50, cosine <0.90, or non-finite",
            },
        },
        "checkpoint_sha256": checkpoint.sha256,
        "baseline_checkpoint_sha256": teacher.sha256,
        "sequence_count": sum(len(sequence) >= 2 for sequence in sequences),
        "prediction_tokens": prediction_count,
        "baseline_perplexity": baseline_perplexity,
        "candidate_float_perplexity": candidate_float_perplexity,
        "baseline_greedy_tokens": list(baseline_sample),
        "evaluations": evaluations,
        "elapsed_seconds": perf_counter() - started,
    }


def write_quantization_report(report: dict, output_prefix: str | Path) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# TinyStories WXAY quantization sweep",
        "",
        f"FP32 teacher-forced perplexity: `{report['baseline_perplexity']:.6f}`  ",
        f"Candidate shadow-FP32 perplexity: `{report.get('candidate_float_perplexity', report['baseline_perplexity']):.6f}`  ",
        f"Prediction tokens: `{report['prediction_tokens']}`  ",
        f"Elapsed: `{report['elapsed_seconds']:.2f} s`",
        f"Weight granularity: `{', '.join(sorted({item['weight_granularity'] for item in report['evaluations']}))}`",
        "",
        "| WXAY | Grade | PPL | PPL ratio | Top-1 agreement | Logit cosine | NRMSE | KL (nats) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["evaluations"]:
        lines.append(
            f"| {item['label']} | {item['grade']} | {item['quantized_perplexity']:.4f} | "
            f"{item['perplexity_ratio']:.4f} | {item['top1_agreement']:.3f} | "
            f"{item['mean_logit_cosine']:.5f} | {item['mean_normalized_rmse']:.5f} | "
            f"{item['mean_kl_nats']:.5f} |"
        )
    lines.extend([
        "",
        "Grades are sensitivity labels for this corpus and fake-quant policy, not final board accuracy claims.",
    ])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
