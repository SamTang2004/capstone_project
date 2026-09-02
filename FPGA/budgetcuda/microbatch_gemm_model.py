"""Screen-and-pack model for sparse micro-computations from tagged GEMVs.

The model separates significance testing from matrix execution. Candidate
micro-blocks are screened, zero-value blocks are dropped, and survivors from
different batch-tagged GEMVs that share a weight tile and K range are packed as
rows of one GEMM launch.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .stories import Llama2CCheckpoint
from .stories_module_goldens import quantize_weight_per_output
from .work_stealing_model import TaskTag


@dataclass(frozen=True)
class MicroCandidate:
    tag: TaskTag
    operation: str
    output_tile: int
    k_start: int
    k_steps: int
    active_k_bits: int

    @property
    def significant(self) -> bool:
        return self.active_k_bits != 0

    @property
    def active_steps(self) -> int:
        return self.active_k_bits.bit_count()

    @property
    def compatibility_key(self) -> tuple[int, int, int, int]:
        return (self.tag.operation_id, self.output_tile, self.k_start, self.k_steps)


@dataclass(frozen=True)
class GemmBundle:
    operation: str
    output_tile: int
    k_start: int
    k_steps: int
    tags: tuple[TaskTag, ...]
    row_active_k_bits: tuple[int, ...]

    @property
    def valid_rows(self) -> int:
        return len(self.tags)

    @property
    def union_active_k_bits(self) -> int:
        value = 0
        for bits in self.row_active_k_bits:
            value |= bits
        return value

    @property
    def compute_steps(self) -> int:
        return self.union_active_k_bits.bit_count()


@dataclass(frozen=True)
class ScreeningCosts:
    predicate_cycles: int = 1
    predicate_lanes: int = 1
    tag_enqueue_cycles: int = 1
    enqueue_lanes: int = 1
    gemm_launch_cycles: int = 2
    tag_route_cycles: int = 0
    matrix_multipliers: int = 2
    gemm_rows: int = 8

    def __post_init__(self) -> None:
        nonnegative = (
            self.predicate_cycles, self.tag_enqueue_cycles,
            self.gemm_launch_cycles, self.tag_route_cycles,
        )
        positive = (
            self.predicate_lanes, self.enqueue_lanes,
            self.matrix_multipliers, self.gemm_rows,
        )
        if any(value < 0 for value in nonnegative) or any(value <= 0 for value in positive):
            raise ValueError("screening costs and machine widths are invalid")


def _checkpoint_matrices(checkpoint: Llama2CCheckpoint) -> Iterable[tuple[str, np.ndarray]]:
    for layer in range(checkpoint.config.n_layers):
        for name in ("wq", "wk", "wv", "wo", "w1", "w3", "w2"):
            yield f"layer_{layer:02d}/{name}", getattr(checkpoint, name)[layer]
    yield "vocabulary_projection", checkpoint.wcls


def _mask_to_bits(mask: np.ndarray) -> int:
    bits = 0
    for index in np.flatnonzero(mask):
        bits |= 1 << int(index)
    return bits


def build_micro_candidates(
    checkpoint: Llama2CCheckpoint,
    *,
    batch_activation_factors: Sequence[float],
    activation_density: float = 0.5,
    significant_code: int = 4,
    k_chunk_steps: int = 16,
    generation: int = 0,
    seed: int = 1,
) -> tuple[MicroCandidate, ...]:
    """Build reproducible activation-dependent micro-block predicates."""

    if not batch_activation_factors or any(
        not math.isfinite(float(value)) or value < 0 for value in batch_activation_factors
    ):
        raise ValueError("batch activation factors must be finite and non-negative")
    if not 0 <= activation_density <= 1:
        raise ValueError("activation_density must be in [0,1]")
    if not 1 <= significant_code <= 7:
        raise ValueError("significant_code must be in 1..7")
    if k_chunk_steps <= 0:
        raise ValueError("k_chunk_steps must be positive")

    rng = np.random.default_rng(seed)
    candidates = []
    for operation_id, (name, weight) in enumerate(_checkpoint_matrices(checkpoint)):
        codes = quantize_weight_per_output(np.asarray(weight, dtype=np.float32), 4).codes
        outputs, reduction = codes.shape
        activation_masks = []
        for factor in batch_activation_factors:
            probability = min(1.0, activation_density * float(factor))
            activation_masks.append(rng.random(reduction) < probability)
        for output_start in range(0, outputs, 8):
            weight_mask = np.any(
                np.abs(codes[output_start:output_start + 8].astype(np.int16)) >= significant_code,
                axis=0,
            )
            output_tile = output_start // 8
            for k_start in range(0, reduction, k_chunk_steps):
                k_stop = min(k_start + k_chunk_steps, reduction)
                local_weight = weight_mask[k_start:k_stop]
                for batch_id, activation_mask in enumerate(activation_masks):
                    active = local_weight & activation_mask[k_start:k_stop]
                    candidates.append(MicroCandidate(
                        TaskTag(batch_id, operation_id, generation),
                        name, output_tile, k_start, k_stop - k_start,
                        _mask_to_bits(active),
                    ))
    return tuple(candidates)


def pack_survivors_as_gemm(
    candidates: Sequence[MicroCandidate], *, gemm_rows: int = 8,
) -> tuple[GemmBundle, ...]:
    if gemm_rows <= 0:
        raise ValueError("gemm_rows must be positive")
    grouped: dict[tuple[int, int, int, int], list[MicroCandidate]] = {}
    for candidate in candidates:
        if candidate.significant:
            grouped.setdefault(candidate.compatibility_key, []).append(candidate)
    bundles = []
    for group in grouped.values():
        for start in range(0, len(group), gemm_rows):
            rows = group[start:start + gemm_rows]
            first = rows[0]
            bundles.append(GemmBundle(
                first.operation, first.output_tile, first.k_start, first.k_steps,
                tuple(row.tag for row in rows),
                tuple(row.active_k_bits for row in rows),
            ))
    return tuple(bundles)


def _list_schedule(cycles: Sequence[int], workers: int) -> int:
    loads = [0] * workers
    for cost in cycles:
        worker = min(range(workers), key=loads.__getitem__)
        loads[worker] += int(cost)
    return max(loads, default=0)


def evaluate_screen_and_pack(
    candidates: Sequence[MicroCandidate], *, costs: ScreeningCosts,
) -> dict:
    if not candidates:
        raise ValueError("at least one micro candidate is required")
    survivors = tuple(candidate for candidate in candidates if candidate.significant)
    bundles = pack_survivors_as_gemm(candidates, gemm_rows=costs.gemm_rows)
    predicate_cycles = math.ceil(len(candidates) / costs.predicate_lanes) * costs.predicate_cycles
    enqueue_cycles = math.ceil(len(survivors) / costs.enqueue_lanes) * costs.tag_enqueue_cycles
    screening_stage_cycles = predicate_cycles + enqueue_cycles
    bundle_costs = [
        costs.gemm_launch_cycles + bundle.compute_steps
        + costs.tag_route_cycles * bundle.valid_rows
        for bundle in bundles
    ]
    matrix_cycles = _list_schedule(bundle_costs, costs.matrix_multipliers)

    keys = {candidate.compatibility_key for candidate in candidates}
    batches = 1 + max(candidate.tag.batch_id for candidate in candidates)
    dense_bundle_costs = []
    for key in keys:
        k_steps = key[3]
        for start in range(0, batches, costs.gemm_rows):
            rows = min(costs.gemm_rows, batches - start)
            dense_bundle_costs.append(
                costs.gemm_launch_cycles + k_steps + costs.tag_route_cycles * rows
            )
    dense_batched_cycles = _list_schedule(dense_bundle_costs, costs.matrix_multipliers)
    dense_gemv_costs = [
        costs.gemm_launch_cycles + key[3] + costs.tag_route_cycles
        for key in keys for _ in range(batches)
    ]
    dense_separate_gemv_cycles = _list_schedule(dense_gemv_costs, costs.matrix_multipliers)
    sparse_separate_costs = [
        costs.gemm_launch_cycles + candidate.active_steps + costs.tag_route_cycles
        for candidate in survivors
    ]
    sparse_separate_matrix_cycles = _list_schedule(
        sparse_separate_costs, costs.matrix_multipliers,
    )
    sparse_separate_serial = screening_stage_cycles + sparse_separate_matrix_cycles
    serialized_cycles = screening_stage_cycles + matrix_cycles
    pipelined_lower_bound = max(screening_stage_cycles, matrix_cycles)

    expected_tags = sorted(
        (candidate.tag, candidate.operation, candidate.output_tile,
         candidate.k_start, candidate.k_steps)
        for candidate in survivors
    )
    packed_tags = sorted(
        (tag, bundle.operation, bundle.output_tile, bundle.k_start, bundle.k_steps)
        for bundle in bundles for tag in bundle.tags
    )
    tag_mismatches = int(expected_tags != packed_tags)
    tag_bits = sum((
        max(1, max(candidate.tag.batch_id for candidate in candidates).bit_length()),
        max(1, max(candidate.tag.operation_id for candidate in candidates).bit_length()),
        max(1, max(candidate.tag.generation for candidate in candidates).bit_length()),
    ))
    useful_row_steps = sum(candidate.active_steps for candidate in survivors)
    issued_row_steps = sum(bundle.compute_steps * bundle.valid_rows for bundle in bundles)
    return {
        "candidate_count": len(candidates),
        "survivor_count": len(survivors),
        "dropped_count": len(candidates) - len(survivors),
        "drop_fraction": 1.0 - len(survivors) / len(candidates),
        "bundle_count": len(bundles),
        "mean_rows_per_bundle": len(survivors) / len(bundles) if bundles else 0.0,
        "gemm_row_utilization": (
            len(survivors) / (len(bundles) * costs.gemm_rows) if bundles else 0.0
        ),
        "useful_active_row_steps": useful_row_steps,
        "issued_row_steps": issued_row_steps,
        "masked_row_step_overhead": issued_row_steps - useful_row_steps,
        "predicate_cycles": predicate_cycles,
        "enqueue_cycles": enqueue_cycles,
        "screening_stage_cycles": screening_stage_cycles,
        "matrix_cycles": matrix_cycles,
        "serialized_cycles": serialized_cycles,
        "pipelined_lower_bound_cycles": pipelined_lower_bound,
        "dense_separate_gemv_cycles": dense_separate_gemv_cycles,
        "dense_batched_gemm_cycles": dense_batched_cycles,
        "screened_sparse_separate_gemv_cycles": sparse_separate_serial,
        "serialized_speedup_vs_dense_separate": dense_separate_gemv_cycles / serialized_cycles,
        "pipelined_speedup_vs_dense_separate": dense_separate_gemv_cycles / pipelined_lower_bound,
        "serialized_speedup_vs_dense_batched": dense_batched_cycles / serialized_cycles,
        "pipelined_speedup_vs_dense_batched": dense_batched_cycles / pipelined_lower_bound,
        "screening_is_pipeline_bottleneck": screening_stage_cycles > matrix_cycles,
        "tag_bits": tag_bits,
        "tag_bytes_per_survivor": math.ceil(tag_bits / 8),
        "tag_metadata_bytes": len(survivors) * math.ceil(tag_bits / 8),
        "tag_mismatches": tag_mismatches,
    }


def evaluate_microbatch_sweep(
    checkpoint: Llama2CCheckpoint,
    *,
    batch_activation_factors: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 1, 1, 1, 1),
    activation_density: float = 0.5,
    significant_code: int = 4,
    chunk_sizes: Sequence[int] = (1, 4, 8, 16, 32, 64),
    predicate_lanes: Sequence[int] = (1, 2, 4, 8, 16),
    predicate_cycles: int = 1,
    tag_enqueue_cycles: int = 1,
    enqueue_lanes: int = 1,
    matrix_multipliers: int = 2,
    gemm_rows: int = 8,
    gemm_launch_cycles: int = 2,
    tag_route_cycles: int = 0,
    seed: int = 1,
) -> dict:
    results = []
    for chunk in chunk_sizes:
        candidates = build_micro_candidates(
            checkpoint,
            batch_activation_factors=batch_activation_factors,
            activation_density=activation_density,
            significant_code=significant_code,
            k_chunk_steps=chunk,
            seed=seed,
        )
        for lanes in predicate_lanes:
            costs = ScreeningCosts(
                predicate_cycles=predicate_cycles,
                predicate_lanes=lanes,
                tag_enqueue_cycles=tag_enqueue_cycles,
                enqueue_lanes=enqueue_lanes,
                gemm_launch_cycles=gemm_launch_cycles,
                tag_route_cycles=tag_route_cycles,
                matrix_multipliers=matrix_multipliers,
                gemm_rows=gemm_rows,
            )
            results.append({
                "k_chunk_steps": chunk,
                "predicate_lanes": lanes,
                **evaluate_screen_and_pack(candidates, costs=costs),
            })
    return {
        "format": "budgetcuda-screen-tag-pack-gemm-model",
        "version": 1,
        "checkpoint_sha256": checkpoint.sha256,
        "assumptions": {
            "batch_activation_factors": [float(value) for value in batch_activation_factors],
            "activation_density": activation_density,
            "significant_w4_code_threshold": significant_code,
            "predicate_cycles": predicate_cycles,
            "tag_enqueue_cycles": tag_enqueue_cycles,
            "enqueue_lanes": enqueue_lanes,
            "matrix_multipliers": matrix_multipliers,
            "gemm_rows": gemm_rows,
            "gemm_launch_cycles": gemm_launch_cycles,
            "tag_route_cycles": tag_route_cycles,
            "activation_masks": "deterministic synthetic Bernoulli masks",
        },
        "results": results,
    }


def write_microbatch_report(report: dict, output_prefix: str | Path) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Screen, tag, and GEMV-to-GEMM validation model",
        "",
        "| K chunk | Predicate lanes | Candidates | Drop | Survivors | Bundles | Rows/bundle | Screen cycles | Matrix cycles | Pipeline bottleneck | Serial speedup vs GEMV | Pipelined speedup vs GEMV | Pipelined speedup vs dense GEMM |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for item in report["results"]:
        lines.append(
            f"| {item['k_chunk_steps']} | {item['predicate_lanes']} | "
            f"{item['candidate_count']} | {item['drop_fraction']:.3f} | "
            f"{item['survivor_count']} | {item['bundle_count']} | "
            f"{item['mean_rows_per_bundle']:.2f} | {item['screening_stage_cycles']} | "
            f"{item['matrix_cycles']} | "
            f"{'screen' if item['screening_is_pipeline_bottleneck'] else 'matrix'} | "
            f"{item['serialized_speedup_vs_dense_separate']:.3f}x | "
            f"{item['pipelined_speedup_vs_dense_separate']:.3f}x | "
            f"{item['pipelined_speedup_vs_dense_batched']:.3f}x |"
        )
    lines.extend([
        "",
        "The pipelined result is the optimistic producer-consumer lower bound max(screen, matrix); the serialized result is the conservative upper bound screen+matrix. Activation masks are synthetic and do not establish an accuracy-safe pruning policy.",
    ])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
