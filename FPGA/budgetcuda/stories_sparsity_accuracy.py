"""Accuracy evaluation for tile-K significance screening in TinyStories W4A8."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np

from .stories import Llama2CCheckpoint, StoriesFloatReference
from .stories_quantization import (StoriesFakeQuantReference,
                                   StoriesQuantizationSpec, _log_softmax)


@dataclass(frozen=True)
class SparsityAccuracySpec:
    significant_w4_code: int = 4
    activation_keep_fraction: float = 0.5
    output_tile: int = 8

    def __post_init__(self) -> None:
        if not 1 <= self.significant_w4_code <= 7:
            raise ValueError("significant_w4_code must be in 1..7")
        if not 0.0 <= self.activation_keep_fraction <= 1.0:
            raise ValueError("activation_keep_fraction must be in [0,1]")
        if self.output_tile <= 0:
            raise ValueError("output_tile must be positive")

    @property
    def label(self) -> str:
        fraction = f"{self.activation_keep_fraction:.3f}".rstrip("0").rstrip(".")
        return f"W4>={self.significant_w4_code}/Akeep={fraction}"


class StoriesScreenedW4A8Reference(StoriesFakeQuantReference):
    """W4A8 reference that discards screened tile-K contributions.

    A K position remains active for an eight-output tile only when at least one
    W4 code in that tile reaches the weight threshold and the input activation
    belongs to the retained magnitude fraction. Once active, all eight W4
    products at that K position are evaluated, matching a row-masked GEMM.
    """

    def __init__(self, checkpoint: Llama2CCheckpoint, policy: SparsityAccuracySpec):
        self.sparsity_policy = policy
        self._screen_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
        self.screened_tile_k = 0
        self.total_tile_k = 0
        super().__init__(checkpoint, StoriesQuantizationSpec(4, 8, "per_output"))

    @staticmethod
    def _weight_key(weight: np.ndarray) -> tuple:
        interface = weight.__array_interface__
        return (int(interface["data"][0]), weight.shape, weight.strides)

    def _screened_weight(self, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = self._weight_key(weight)
        cached = self._screen_cache.get(key)
        if cached is not None:
            return cached
        weight = np.asarray(weight, dtype=np.float32)
        maximum = np.max(np.abs(weight), axis=1, keepdims=True)
        scale = np.where(maximum == 0, np.float32(1.0), maximum / np.float32(7.0))
        codes = np.clip(np.rint(weight / scale), -7, 7).astype(np.int8)
        outputs, reduction = codes.shape
        tiles = math.ceil(outputs / self.sparsity_policy.output_tile)
        tile_mask = np.zeros((tiles, reduction), dtype=bool)
        expanded = np.zeros_like(codes, dtype=bool)
        for tile, output_start in enumerate(
            range(0, outputs, self.sparsity_policy.output_tile)
        ):
            output_stop = min(output_start + self.sparsity_policy.output_tile, outputs)
            active = np.any(
                np.abs(codes[output_start:output_stop].astype(np.int16))
                >= self.sparsity_policy.significant_w4_code,
                axis=0,
            )
            tile_mask[tile] = active
            expanded[output_start:output_stop] = active
        result = (np.where(expanded, weight, np.float32(0.0)), tile_mask)
        self._screen_cache[key] = result
        return result

    def _activation_mask(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        keep = self.sparsity_policy.activation_keep_fraction
        count = int(math.ceil(value.size * keep))
        if count <= 0:
            return np.zeros(value.size, dtype=bool)
        if count >= value.size:
            return np.ones(value.size, dtype=bool)
        indices = np.argpartition(np.abs(value), value.size - count)[value.size - count:]
        mask = np.zeros(value.size, dtype=bool)
        mask[indices] = True
        return mask

    def _matmul(self, weight: np.ndarray, value: np.ndarray) -> np.ndarray:
        screened_weight, tile_mask = self._screened_weight(weight)
        activation_mask = self._activation_mask(value)
        self.total_tile_k += int(tile_mask.size)
        self.screened_tile_k += int(tile_mask.size - np.count_nonzero(
            tile_mask & activation_mask[np.newaxis, :]
        ))
        screened_value = np.where(activation_mask, value, np.float32(0.0))
        return np.asarray(screened_weight @ screened_value, dtype=np.float32)

    @property
    def measured_drop_fraction(self) -> float:
        return self.screened_tile_k / self.total_tile_k if self.total_tile_k else 0.0


def _run_logits(reference, sequences: Sequence[Sequence[int]]):
    all_logits = []
    targets = []
    for sequence in sequences:
        if len(sequence) < 2:
            continue
        results = reference.run_tokens(sequence[:-1], capture=False)
        all_logits.append(tuple(np.asarray(item.logits, dtype=np.float32) for item in results))
        targets.append(tuple(int(token) for token in sequence[1:]))
    if not targets:
        raise ValueError("evaluation requires at least one next-token target")
    return all_logits, targets


def _metrics(candidate, baseline, targets) -> dict:
    count = 0
    nll = 0.0
    top1 = 0
    cosine = 0.0
    nrmse = 0.0
    kl = 0.0
    for candidate_sequence, baseline_sequence, sequence_targets in zip(
        candidate, baseline, targets
    ):
        for value, base, target in zip(
            candidate_sequence, baseline_sequence, sequence_targets
        ):
            value64 = np.asarray(value, dtype=np.float64)
            base64 = np.asarray(base, dtype=np.float64)
            value_logp = _log_softmax(value64)
            base_logp = _log_softmax(base64)
            nll -= float(value_logp[target])
            top1 += int(np.argmax(value64) == np.argmax(base64))
            denominator = float(np.linalg.norm(value64) * np.linalg.norm(base64))
            cosine += float(np.dot(value64, base64) / denominator) if denominator else 0.0
            base_norm = float(np.linalg.norm(base64))
            nrmse += float(np.linalg.norm(value64 - base64) / base_norm) if base_norm else 0.0
            probability = np.exp(base_logp)
            kl += float(np.sum(probability * (base_logp - value_logp)))
            count += 1
    average_nll = nll / count
    return {
        "prediction_tokens": count,
        "perplexity": math.exp(average_nll) if average_nll < 700 else math.inf,
        "top1_agreement": top1 / count,
        "mean_logit_cosine": cosine / count,
        "mean_normalized_rmse": nrmse / count,
        "mean_kl_nats": kl / count,
    }


def evaluate_sparsity_accuracy(
    checkpoint: Llama2CCheckpoint,
    sequences: Sequence[Sequence[int]],
    policies: Sequence[SparsityAccuracySpec],
    *,
    baseline_checkpoint: Llama2CCheckpoint | None = None,
) -> dict:
    """Measure incremental screening loss against dense W4A8 and FP32."""

    started = perf_counter()
    teacher = baseline_checkpoint or checkpoint
    if teacher.config != checkpoint.config:
        raise ValueError("baseline and candidate checkpoint configs must match")
    teacher_logits, targets = _run_logits(StoriesFloatReference(teacher), sequences)
    dense_logits, dense_targets = _run_logits(
        StoriesFakeQuantReference(
            checkpoint, StoriesQuantizationSpec(4, 8, "per_output")
        ),
        sequences,
    )
    if dense_targets != targets:
        raise AssertionError("dense and teacher targets differ")
    teacher_self = _metrics(teacher_logits, teacher_logits, targets)
    dense_vs_teacher = _metrics(dense_logits, teacher_logits, targets)
    evaluations = []
    for policy in policies:
        policy_started = perf_counter()
        reference = StoriesScreenedW4A8Reference(checkpoint, policy)
        sparse_logits, sparse_targets = _run_logits(reference, sequences)
        if sparse_targets != targets:
            raise AssertionError("sparse and teacher targets differ")
        versus_teacher = _metrics(sparse_logits, teacher_logits, targets)
        versus_dense = _metrics(sparse_logits, dense_logits, targets)
        evaluations.append({
            "label": policy.label,
            "significant_w4_code": policy.significant_w4_code,
            "activation_keep_fraction": policy.activation_keep_fraction,
            "measured_tile_k_drop_fraction": reference.measured_drop_fraction,
            "perplexity": versus_teacher["perplexity"],
            "perplexity_ratio_vs_fp32": (
                versus_teacher["perplexity"] / teacher_self["perplexity"]
            ),
            "perplexity_ratio_vs_dense_w4a8": (
                versus_teacher["perplexity"] / dense_vs_teacher["perplexity"]
            ),
            "vs_fp32": versus_teacher,
            "vs_dense_w4a8": versus_dense,
            "elapsed_seconds": perf_counter() - policy_started,
        })
    return {
        "format": "budgetcuda-stories-screened-w4a8-accuracy",
        "version": 1,
        "checkpoint_sha256": checkpoint.sha256,
        "baseline_checkpoint_sha256": teacher.sha256,
        "sequence_count": len(targets),
        "prediction_tokens": teacher_self["prediction_tokens"],
        "fp32_perplexity": teacher_self["perplexity"],
        "dense_w4a8": dense_vs_teacher,
        "evaluations": evaluations,
        "elapsed_seconds": perf_counter() - started,
    }


def write_sparsity_accuracy_report(report: dict, output_prefix: str | Path):
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    dense = report["dense_w4a8"]
    lines = [
        "# TinyStories screened W4A8 accuracy",
        "",
        f"FP32 perplexity: `{report['fp32_perplexity']:.6f}`  ",
        f"Dense W4A8 perplexity: `{dense['perplexity']:.6f}`  ",
        f"Prediction tokens: `{report['prediction_tokens']}`",
        "",
        "| Policy | Tile-K dropped | PPL | PPL / FP32 | PPL / dense W4A8 | FP32 top-1 | Dense W4A8 top-1 | FP32 cosine | KL vs dense W4A8 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["evaluations"]:
        lines.append(
            f"| {item['label']} | {item['measured_tile_k_drop_fraction']:.1%} | "
            f"{item['perplexity']:.4f} | {item['perplexity_ratio_vs_fp32']:.4f} | "
            f"{item['perplexity_ratio_vs_dense_w4a8']:.4f} | "
            f"{item['vs_fp32']['top1_agreement']:.3f} | "
            f"{item['vs_dense_w4a8']['top1_agreement']:.3f} | "
            f"{item['vs_fp32']['mean_logit_cosine']:.5f} | "
            f"{item['vs_dense_w4a8']['mean_kl_nats']:.5f} |"
        )
    lines.extend([
        "",
        "Activation retention uses the largest-magnitude real A8 values in each GEMV. Results measure inference pruning; they do not reuse the synthetic Bernoulli masks from the scheduler-throughput model.",
    ])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
