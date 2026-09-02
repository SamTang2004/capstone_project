"""Discrete-event scheduler model for sparse TinyStories W4 GEMM work.

The model keeps transformer GEMM dependency barriers intact and exposes the
trade-off between K-chunk granularity, load imbalance, descriptor traffic, and
scheduler/steal overhead. It is intentionally architectural rather than cycle
accurate: every active K step costs one GEMM-core cycle and all control costs
are explicit parameters.
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .stories import Llama2CCheckpoint
from .stories_module_goldens import quantize_weight_per_output


@dataclass(frozen=True)
class TileTask:
    operation: str
    output_tile: int
    k_start: int
    k_steps: int
    active_steps: int

    @property
    def compute_cycles(self) -> int:
        return self.active_steps


@dataclass(frozen=True)
class WorkloadOperation:
    name: str
    reduction: int
    outputs: int
    dense_cycles_single_core: int
    active_cycles: int
    tasks: tuple[TileTask, ...]


@dataclass(frozen=True)
class SchedulerCosts:
    task_dispatch_cycles: int = 1
    steal_probe_cycles: int = 4
    steal_transfer_cycles: int = 4
    partial_commit_cycles: int = 1
    operation_barrier_cycles: int = 2
    descriptor_bytes: int = 16
    steal_batch: int = 4

    def __post_init__(self) -> None:
        numeric = (
            self.task_dispatch_cycles, self.steal_probe_cycles,
            self.steal_transfer_cycles, self.partial_commit_cycles,
            self.operation_barrier_cycles, self.descriptor_bytes,
        )
        if any(value < 0 for value in numeric) or self.steal_batch <= 0:
            raise ValueError("scheduler costs must be non-negative and steal_batch positive")


@dataclass
class ScheduleMetrics:
    policy: str
    makespan_cycles: int = 0
    compute_cycles: int = 0
    scheduler_cycles: int = 0
    commit_cycles: int = 0
    barrier_cycles: int = 0
    task_count: int = 0
    steal_count: int = 0
    stolen_tasks: int = 0
    descriptor_bytes: int = 0
    worker_count: int = 0

    @property
    def utilization(self) -> float:
        denominator = self.worker_count * self.makespan_cycles
        return self.compute_cycles / denominator if denominator else 0.0

    @property
    def aggregate_overhead_fraction(self) -> float:
        total = self.compute_cycles + self.scheduler_cycles + self.commit_cycles
        return (self.scheduler_cycles + self.commit_cycles) / total if total else 0.0

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "makespan_cycles": self.makespan_cycles,
            "compute_cycles": self.compute_cycles,
            "scheduler_cycles": self.scheduler_cycles,
            "commit_cycles": self.commit_cycles,
            "barrier_cycles": self.barrier_cycles,
            "task_count": self.task_count,
            "steal_count": self.steal_count,
            "stolen_tasks": self.stolen_tasks,
            "descriptor_bytes": self.descriptor_bytes,
            "worker_count": self.worker_count,
            "utilization": self.utilization,
            "aggregate_overhead_fraction": self.aggregate_overhead_fraction,
        }


def _checkpoint_matrices(checkpoint: Llama2CCheckpoint) -> Iterable[tuple[str, np.ndarray]]:
    for layer in range(checkpoint.config.n_layers):
        for name in ("wq", "wk", "wv", "wo", "w1", "w3", "w2"):
            yield f"layer_{layer:02d}/{name}", getattr(checkpoint, name)[layer]
    yield "vocabulary_projection", checkpoint.wcls


def build_stories_w4_workload(
    checkpoint: Llama2CCheckpoint,
    *,
    significant_code: int = 4,
    k_chunk_steps: int | None = None,
) -> tuple[WorkloadOperation, ...]:
    """Build tasks from real per-output W4 codes in the QAT checkpoint."""

    if significant_code < 1 or significant_code > 7:
        raise ValueError("significant_code must be in 1..7")
    if k_chunk_steps is not None and k_chunk_steps <= 0:
        raise ValueError("k_chunk_steps must be positive or None")
    operations = []
    for name, weight in _checkpoint_matrices(checkpoint):
        weight = np.asarray(weight, dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError(f"matrix {name} is not rank two")
        outputs, reduction = weight.shape
        codes = quantize_weight_per_output(weight, 4).codes
        tasks = []
        active_total = 0
        chunk = reduction if k_chunk_steps is None else min(k_chunk_steps, reduction)
        for output_start in range(0, outputs, 8):
            tile_codes = codes[output_start:output_start + 8]
            significant = np.any(np.abs(tile_codes.astype(np.int16)) >= significant_code, axis=0)
            output_tile = output_start // 8
            for k_start in range(0, reduction, chunk):
                k_stop = min(k_start + chunk, reduction)
                active = int(np.count_nonzero(significant[k_start:k_stop]))
                if active:
                    tasks.append(TileTask(
                        name, output_tile, k_start, k_stop - k_start, active,
                    ))
                    active_total += active
        dense_cycles = math.ceil(outputs / 8) * reduction
        operations.append(WorkloadOperation(
            name, reduction, outputs, dense_cycles, active_total, tuple(tasks),
        ))
    return tuple(operations)


def _ordered_tasks(
    tasks: Sequence[TileTask], scenario: str, workers: int, seed: int,
) -> list[TileTask]:
    tasks = list(tasks)
    if scenario == "actual":
        return tasks
    if scenario == "shuffled":
        random.Random(seed).shuffle(tasks)
        return tasks
    if scenario == "pathological":
        # Round-robin assignment sends the lightest bucket to worker 0 and the
        # heaviest to the last worker, reproducing the motivating imbalance.
        ordered = sorted(tasks, key=lambda task: task.compute_cycles)
        bucket_size = math.ceil(len(ordered) / workers)
        buckets = [
            ordered[index * bucket_size:(index + 1) * bucket_size]
            for index in range(workers)
        ]
        buckets.sort(key=lambda bucket: sum(task.compute_cycles for task in bucket))
        result = []
        for index in range(max((len(bucket) for bucket in buckets), default=0)):
            for bucket in buckets:
                if index < len(bucket):
                    result.append(bucket[index])
        return result
    raise ValueError("scenario must be actual, shuffled, or pathological")


def _merge_metrics(total: ScheduleMetrics, current: ScheduleMetrics) -> None:
    for name in (
        "makespan_cycles", "compute_cycles", "scheduler_cycles", "commit_cycles",
        "barrier_cycles", "task_count", "steal_count", "stolen_tasks",
        "descriptor_bytes",
    ):
        setattr(total, name, getattr(total, name) + getattr(current, name))


def schedule_static(
    operations: Sequence[WorkloadOperation],
    *,
    workers: int,
    costs: SchedulerCosts,
    scenario: str = "actual",
    seed: int = 1,
) -> ScheduleMetrics:
    if workers <= 0:
        raise ValueError("workers must be positive")
    total = ScheduleMetrics("static_round_robin", worker_count=workers)
    for operation_index, operation in enumerate(operations):
        loads = [0] * workers
        tasks = _ordered_tasks(operation.tasks, scenario, workers, seed + operation_index)
        for index, task in enumerate(tasks):
            loads[index % workers] += (
                costs.task_dispatch_cycles + task.compute_cycles + costs.partial_commit_cycles
            )
        current = ScheduleMetrics(
            "static_round_robin",
            makespan_cycles=max(loads, default=0) + costs.operation_barrier_cycles,
            compute_cycles=sum(task.compute_cycles for task in tasks),
            scheduler_cycles=len(tasks) * costs.task_dispatch_cycles,
            commit_cycles=len(tasks) * costs.partial_commit_cycles,
            barrier_cycles=costs.operation_barrier_cycles,
            task_count=len(tasks),
            descriptor_bytes=len(tasks) * costs.descriptor_bytes,
            worker_count=workers,
        )
        _merge_metrics(total, current)
    return total


def schedule_central_queue(
    operations: Sequence[WorkloadOperation],
    *,
    workers: int,
    costs: SchedulerCosts,
    scenario: str = "actual",
    seed: int = 1,
) -> ScheduleMetrics:
    total = ScheduleMetrics("central_queue", worker_count=workers)
    for operation_index, operation in enumerate(operations):
        available = [0] * workers
        dispatcher_available = 0
        tasks = _ordered_tasks(operation.tasks, scenario, workers, seed + operation_index)
        for task in tasks:
            worker = min(range(workers), key=lambda index: available[index])
            dispatch_start = max(available[worker], dispatcher_available)
            dispatch_done = dispatch_start + costs.task_dispatch_cycles
            dispatcher_available = dispatch_done
            available[worker] = dispatch_done + task.compute_cycles + costs.partial_commit_cycles
        current = ScheduleMetrics(
            "central_queue",
            makespan_cycles=max(available, default=0) + costs.operation_barrier_cycles,
            compute_cycles=sum(task.compute_cycles for task in tasks),
            scheduler_cycles=len(tasks) * costs.task_dispatch_cycles,
            commit_cycles=len(tasks) * costs.partial_commit_cycles,
            barrier_cycles=costs.operation_barrier_cycles,
            task_count=len(tasks),
            descriptor_bytes=len(tasks) * costs.descriptor_bytes,
            worker_count=workers,
        )
        _merge_metrics(total, current)
    return total


def schedule_work_stealing(
    operations: Sequence[WorkloadOperation],
    *,
    workers: int,
    costs: SchedulerCosts,
    scenario: str = "actual",
    seed: int = 1,
) -> ScheduleMetrics:
    total = ScheduleMetrics("work_stealing", worker_count=workers)
    for operation_index, operation in enumerate(operations):
        tasks = _ordered_tasks(operation.tasks, scenario, workers, seed + operation_index)
        queues = [deque() for _ in range(workers)]
        for index, task in enumerate(tasks):
            queues[index % workers].append(task)
        remaining_work = [
            sum(task.compute_cycles for task in queue) for queue in queues
        ]
        available = [0] * workers
        scheduler_cycles = 0
        steal_count = 0
        stolen_tasks = 0
        while any(queues):
            worker = min(range(workers), key=lambda index: available[index])
            if not queues[worker]:
                victims = [index for index in range(workers) if index != worker and queues[index]]
                if not victims:
                    break
                victim = max(
                    victims,
                    key=lambda index: remaining_work[index],
                )
                steal_cost = costs.steal_probe_cycles + costs.steal_transfer_cycles
                if (len(queues[victim]) == 1
                        and available[worker] + steal_cost >= available[victim]):
                    # Let the owner launch its last queued task. Stealing it
                    # would make the two idle workers exchange the same task.
                    worker = victim
                else:
                    transfer = min(
                        costs.steal_batch,
                        max(1, len(queues[victim]) // 2),
                    )
                    available[worker] += steal_cost
                    scheduler_cycles += steal_cost
                    stolen = [queues[victim].pop() for _ in range(transfer)]
                    queues[worker].extend(reversed(stolen))
                    transferred_work = sum(task.compute_cycles for task in stolen)
                    remaining_work[victim] -= transferred_work
                    remaining_work[worker] += transferred_work
                    steal_count += 1
                    stolen_tasks += transfer
                    continue
            task = queues[worker].popleft()
            remaining_work[worker] -= task.compute_cycles
            available[worker] += (
                costs.task_dispatch_cycles + task.compute_cycles + costs.partial_commit_cycles
            )
            scheduler_cycles += costs.task_dispatch_cycles
        current = ScheduleMetrics(
            "work_stealing",
            makespan_cycles=max(available, default=0) + costs.operation_barrier_cycles,
            compute_cycles=sum(task.compute_cycles for task in tasks),
            scheduler_cycles=scheduler_cycles,
            commit_cycles=len(tasks) * costs.partial_commit_cycles,
            barrier_cycles=costs.operation_barrier_cycles,
            task_count=len(tasks),
            steal_count=steal_count,
            stolen_tasks=stolen_tasks,
            descriptor_bytes=len(tasks) * costs.descriptor_bytes,
            worker_count=workers,
        )
        _merge_metrics(total, current)
    return total


def dense_baselines(
    operations: Sequence[WorkloadOperation], workers: int,
) -> dict[str, int]:
    single = sum(operation.dense_cycles_single_core for operation in operations)
    split_n = sum(
        math.ceil(math.ceil(operation.outputs / 8) / workers) * operation.reduction
        for operation in operations
    )
    sparse_ideal = sum(
        math.ceil(operation.active_cycles / workers) for operation in operations
    )
    return {
        "dense_single_core_cycles": single,
        "dense_n_tile_split_cycles": split_n,
        "sparse_dependency_aware_ideal_cycles": sparse_ideal,
    }


def find_steal_latency_break_even(
    operations: Sequence[WorkloadOperation],
    *,
    workers: int,
    static: ScheduleMetrics,
    scenario: str,
    base_costs: SchedulerCosts,
    maximum_total_steal_cycles: int = 4096,
) -> int | None:
    """Find the contiguous steal-latency budget before the first loss.

    Local task dispatch remains fixed at ``base_costs.task_dispatch_cycles``;
    only probe-plus-transfer latency changes. Very large latency can change the
    steal decisions and make the model non-monotonic, so this deliberately
    returns the conservative winning interval connected to zero latency.
    """

    def wins(total: int) -> bool:
        costs = replace(
            base_costs,
            steal_probe_cycles=total // 2,
            steal_transfer_cycles=total - total // 2,
        )
        stealing = schedule_work_stealing(
            operations, workers=workers, costs=costs, scenario=scenario,
        )
        return stealing.makespan_cycles < static.makespan_cycles

    if not wins(0):
        return None
    for total in range(1, maximum_total_steal_cycles + 1):
        if not wins(total):
            return total - 1
    return maximum_total_steal_cycles


def evaluate_scheduler_overhead(
    checkpoint: Llama2CCheckpoint,
    *,
    workers: int = 2,
    significant_code: int = 4,
    chunk_sizes: Sequence[int | None] = (None, 64, 32, 16),
    overhead_values: Sequence[int] = (0, 1, 2, 4, 8, 16, 32, 64),
    scenarios: Sequence[str] = ("actual", "pathological"),
    descriptor_bytes: int = 16,
    partial_commit_cycles: int = 1,
    operation_barrier_cycles: int = 2,
    steal_batch: int = 4,
) -> dict:
    results = []
    for chunk in chunk_sizes:
        operations = build_stories_w4_workload(
            checkpoint, significant_code=significant_code, k_chunk_steps=chunk,
        )
        task_count = sum(len(operation.tasks) for operation in operations)
        active_cycles = sum(operation.active_cycles for operation in operations)
        dense = dense_baselines(operations, workers)
        for scenario in scenarios:
            samples = []
            static_costs = SchedulerCosts(
                task_dispatch_cycles=1,
                partial_commit_cycles=partial_commit_cycles,
                operation_barrier_cycles=operation_barrier_cycles,
                descriptor_bytes=descriptor_bytes,
                steal_batch=steal_batch,
            )
            static = schedule_static(
                operations, workers=workers, costs=static_costs, scenario=scenario,
            )
            steal_latency_break_even = find_steal_latency_break_even(
                operations, workers=workers, static=static, scenario=scenario,
                base_costs=static_costs,
            )
            recommended_costs = replace(
                static_costs, steal_probe_cycles=4, steal_transfer_cycles=4,
            )
            recommended = schedule_work_stealing(
                operations, workers=workers, costs=recommended_costs,
                scenario=scenario,
            )
            break_even = None
            for overhead in overhead_values:
                costs = SchedulerCosts(
                    task_dispatch_cycles=overhead,
                    steal_probe_cycles=2 * overhead + 1,
                    steal_transfer_cycles=2 * overhead + 1,
                    partial_commit_cycles=partial_commit_cycles,
                    operation_barrier_cycles=operation_barrier_cycles,
                    descriptor_bytes=descriptor_bytes,
                    steal_batch=steal_batch,
                )
                central = schedule_central_queue(
                    operations, workers=workers, costs=costs, scenario=scenario,
                )
                stealing = schedule_work_stealing(
                    operations, workers=workers, costs=costs, scenario=scenario,
                )
                if stealing.makespan_cycles < static.makespan_cycles:
                    break_even = overhead
                samples.append({
                    "task_overhead_cycles": overhead,
                    "central_queue": central.as_dict(),
                    "work_stealing": stealing.as_dict(),
                    "work_stealing_speedup_vs_static": (
                        static.makespan_cycles / stealing.makespan_cycles
                        if stealing.makespan_cycles else math.inf
                    ),
                })
            results.append({
                "k_chunk_steps": chunk,
                "scenario": scenario,
                "operation_count": len(operations),
                "task_count": task_count,
                "descriptor_bytes": task_count * descriptor_bytes,
                "active_cycles": active_cycles,
                "active_fraction_of_dense": active_cycles / dense["dense_single_core_cycles"],
                "dense_baselines": dense,
                "static": static.as_dict(),
                "work_stealing_local1_steal8": recommended.as_dict(),
                "work_stealing_local1_steal8_speedup_vs_static": (
                    static.makespan_cycles / recommended.makespan_cycles
                    if recommended.makespan_cycles else math.inf
                ),
                "max_sampled_break_even_task_overhead_cycles": break_even,
                "contiguous_break_even_total_steal_latency_at_one_cycle_dispatch": steal_latency_break_even,
                "samples": samples,
            })
    return {
        "format": "budgetcuda-work-stealing-overhead-model",
        "version": 1,
        "checkpoint_sha256": checkpoint.sha256,
        "assumptions": {
            "workers": workers,
            "significant_w4_code_threshold": significant_code,
            "one_active_k_step_compute_cycles": 1,
            "descriptor_bytes": descriptor_bytes,
            "partial_commit_cycles": partial_commit_cycles,
            "operation_barrier_cycles": operation_barrier_cycles,
            "steal_batch": steal_batch,
            "steal_cost_formula": "probe=2*task_overhead+1; transfer=2*task_overhead+1",
            "static_task_dispatch_cycles": 1,
        },
        "results": results,
    }


def write_scheduler_report(report: dict, output_prefix: str | Path) -> tuple[Path, Path]:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# TinyStories sparse work-stealing overhead model",
        "",
        "| K chunk | Scenario | Tasks | Active/dense | Descriptor KiB | Static cycles | WS cycles 1/8* | Speedup | WS util. | Steals | Task dispatch break-even | Steal latency break-even† |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in report["results"]:
        stealing = result["work_stealing_local1_steal8"]
        chunk = "whole" if result["k_chunk_steps"] is None else result["k_chunk_steps"]
        break_even = result["max_sampled_break_even_task_overhead_cycles"]
        lines.append(
            f"| {chunk} | {result['scenario']} | {result['task_count']} | "
            f"{result['active_fraction_of_dense']:.3f} | {result['descriptor_bytes'] / 1024:.1f} | "
            f"{result['static']['makespan_cycles']} | {stealing['makespan_cycles']} | "
            f"{result['work_stealing_local1_steal8_speedup_vs_static']:.3f}x | {stealing['utilization']:.3f} | "
            f"{stealing['steal_count']} | {break_even if break_even is not None else 'none'} | "
            f"{result['contiguous_break_even_total_steal_latency_at_one_cycle_dispatch']} |"
        )
    lines.extend([
        "",
        "*The recommended comparison uses one cycle per local task dispatch and eight total cycles per steal handshake. The task-dispatch break-even is the largest tested per-task overhead for which work stealing beats static round-robin. †Steal-latency break-even holds local dispatch at one cycle and reports the conservative winning interval connected to zero probe-plus-transfer latency. These are architectural model bounds, not cycle-accurate RTL claims.",
    ])
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
