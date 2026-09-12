"""Synthetic ready-wave experiment; not measured FPGA performance.

Two independent hypothetical workers, one serialized admission controller and
one serialized operand-migration link. Local operands are pinned. Local memory
wait/output service is private; migration transfers inputs/descriptor only.
No dependencies, eviction, shared DDR or numerical sparsification is simulated.
"""
from dataclasses import dataclass
import argparse
import heapq
import json
from pathlib import Path


@dataclass(frozen=True)
class Work:
    uid: int
    batch: int
    owner: int
    compute: int
    migration: int
    wait: int = 0
    lanes: int = 64


def quantize(cycles, t0):
    return (cycles + t0 - 1) // t0 * t0


def simulate(tasks, policy="fifo", t0=1, probe=1, padding=False):
    """One probe per idle event; rejected workers wake on a later completion.

    fifo: local FIFO only; naive: steal tail from longest queue;
    aware: scan queued candidates, steal only if predicted wave end improves.
    Estimation uses T0-rounded services; actual cycles are unchanged unless
    padding=True. Admission scans cost probe cycles PER candidate, including
    rejected candidates. Active work is never moved. All policies pay one
    dispatch cycle per task. Heap tie order is deterministic worker ID.
    Accepted candidates are atomically reserved at the probe snapshot; decision
    latency delays the thief, not the victim. Real arbitration must implement
    reservation/version checking. Costs are exact synthetic inputs (no predictor
    error). T0 does not make the modeled comparator cheaper automatically.
    """
    if policy not in {"fifo", "naive", "aware"} or t0 < 1 or probe < 0:
        raise ValueError("invalid policy/cost")
    if len({t.uid for t in tasks}) != len(tasks):
        raise ValueError("duplicate task")
    if any(t.owner not in (0, 1) or t.compute <= 0 or min(t.migration, t.wait) < 0
           or not 1 <= t.lanes <= 64 for t in tasks):
        raise ValueError("invalid work")
    queues = [[t for t in tasks if t.owner == w] for w in range(2)]
    ready = [0, 0]
    events = [(0, 0), (0, 1)]
    controller = link = 0
    trace = []
    completed = set()
    counts = dict(steals=0, rejected_probes=0, probe_cycles=0,
                  migration_cycles=0, padding_cycles=0)

    def actual(t):
        raw = t.compute + t.wait + 1
        return quantize(raw, t0) if padding else raw

    def estimate(t):
        return quantize(t.compute + t.wait + 1, t0)

    while events:
        now, worker = heapq.heappop(events)
        if now != ready[worker]:
            continue
        task = None
        start = now
        if queues[worker]:
            task = queues[worker].pop(0)
        elif policy != "fifo":
            victim = 1 - worker
            candidates = list(queues[victim])
            if candidates:
                if policy == "naive":
                    candidates = candidates[-1:]
                cost = probe * len(candidates)
                decision = max(now, controller) + cost
                controller = decision
                counts["probe_cycles"] += cost
                baseline = max(now, ready[victim]) + sum(map(estimate, queues[victim]))
                options = []
                for candidate in candidates:
                    remote = max(decision, link) + quantize(candidate.migration, t0) + estimate(candidate)
                    local = max(now, ready[victim]) + sum(map(estimate, queues[victim])) - estimate(candidate)
                    options.append((max(remote, local), candidate.uid, candidate))
                predicted, _, chosen = min(options)
                if policy == "naive" or predicted < baseline:
                    task = chosen
                    queues[victim].remove(task)
                    transfer = quantize(task.migration, t0) if padding else task.migration
                    start = max(decision, link) + transfer
                    link = start
                    counts["migration_cycles"] += transfer
                    counts["steals"] += 1
                else:
                    counts["rejected_probes"] += 1
                    # Probe cost is real even when no work is admitted.
                    ready[worker] = decision
                    if queues[victim] and ready[victim] > now:
                        wake = max(decision, ready[victim])
                        ready[worker] = wake
                        heapq.heappush(events, (wake, worker))
                    continue
        if task is None:
            # Reconsider on a peer completion, without unbounded polling.
            other = 1 - worker
            if policy != "fifo" and queues[other] and ready[other] > now:
                ready[worker] = ready[other]
                heapq.heappush(events, (ready[worker], worker))
            continue
        finish = start + actual(task)
        counts["padding_cycles"] += actual(task) - task.compute - task.wait - 1
        ready[worker] = finish
        trace.append(dict(uid=task.uid, batch=task.batch, owner=task.owner,
                          worker=worker, start=start, finish=finish))
        assert task.uid not in completed
        completed.add(task.uid)
        heapq.heappush(events, (finish, worker))
    assert completed == {t.uid for t in tasks}
    makespan = max(ready)
    compute = sum(t.compute for t in tasks)
    useful = sum(t.compute * t.lanes for t in tasks)
    return dict(policy=policy, t0=t0, probe=probe, padding=padding,
                makespan=makespan, compute_cycles=compute,
                compute_util=compute / (2 * makespan) if makespan else 0,
                useful_lane_util=useful / (128 * makespan) if makespan else 0,
                **counts, trace=trace)


def workloads():
    return {
        "balanced": [Work(i, i % 2, i % 2, 64, 8) for i in range(24)],
        "skew_resident": [Work(i, i % 3, 0 if i < 20 else 1, 64, 8) for i in range(24)],
        "skew_cold": [Work(i, i % 3, 0 if i < 20 else 1, 64, 2048) for i in range(24)],
        "mixed_cost": [Work(i, i % 3, 0 if i < 20 else 1,
                            (8, 32, 64, 128)[i % 4], 2048 if i % 3 == 0 else 8)
                       for i in range(24)],
        "tiny": [Work(i, i % 3, 0 if i < 100 else 1, 2, 4) for i in range(120)],
        "gemv_skew": [Work(i, i % 3, 0 if i < 20 else 1, 64, 8, lanes=8) for i in range(24)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name, tasks in workloads().items():
        baseline = simulate(tasks)
        configs = [("fifo", 1, 1, False), ("naive", 1, 1, False)]
        configs += [("aware", t, p, pad) for p in (0, 1, 4, 16)
                    for t in (1, 4, 16, 64, 128) for pad in (False, True)]
        for policy, t, p, pad in configs:
            row = simulate(tasks, policy, t, p, pad)
            row.pop("trace")
            row.update(scenario=name, speedup=baseline["makespan"] / row["makespan"])
            matched = simulate(tasks, t0=t, padding=pad)
            row['speedup_vs_same_padding_fifo'] = matched['makespan'] / row['makespan']
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(synthetic=True, rows=rows), indent=2))
    for r in rows:
        if r["policy"] != "aware" or (r["probe"] == 1 and r["t0"] in (1, 64)):
            print(f'{r["scenario"]:14} {r["policy"]:5} T0={r["t0"]:3} pad={str(r["padding"]):5} '
                  f'cycles={r["makespan"]:5} speedup={r["speedup"]:.3f} '
                  f'compute={r["compute_util"]:.3f} lanes={r["useful_lane_util"]:.3f}')


if __name__ == "__main__":
    main()
