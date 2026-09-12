"""Pre-wave W4A8 row fusion reference, not an RTL latency model.

T0 is one resident 8x8xK0 completion. Work is regrouped before issue; no running
tile is stolen. Each job is one GEMV with an already-ready activation vector.
"""
from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class Job:
    batch: int
    task: int
    weight_id: str
    activation: tuple


def plan(jobs, weights, *, k0=16, t0=64, s=32, threshold=0,
         checks_per_cycle=8, pack_bytes_per_cycle=8, metadata_cycles=2):
    """weights[id] is signed W4 [K][N], N<=8, common integer scale domain.

    Offline weight max-abs per K is presumed available. One online check tests
    abs(A[k])*max_abs(B[k]) <= threshold; it drops that ROW's entire K vector.
    threshold=0 is exact. Other thresholds return accumulated output-error bounds.
    Only identical weight_id and batch share tiles. Plan cost includes scan,
    grouping metadata, packed A/B writes, and a simple per-output accumulation
    charge for fragments. No DMA read cost/predictor uncertainty is modeled.
    Cost is an analytical serial-stage budget, NOT Python elapsed time.
    """
    if min(k0, t0, checks_per_cycle, pack_bytes_per_cycle) <= 0 or k0 % 2:
        raise ValueError('positive costs and even K0 required for W4 alignment')
    if s < 0 or threshold < 0 or metadata_cycles < 0:
        raise ValueError('negative budget/cost')
    if len({(j.batch, j.task) for j in jobs}) != len(jobs):
        raise ValueError('duplicate tag')
    for j in jobs:
        b = weights[j.weight_id]
        if not b or not 1 <= len(b[0]) <= 8 or len(j.activation) != len(b):
            raise ValueError('invalid shape')
        if any(len(row) != len(b[0]) or any(not -8 <= x <= 7 for x in row) for row in b):
            raise ValueError('weights must be rectangular W4')
        if any(not -128 <= a <= 127 for a in j.activation):
            raise ValueError('activations must be A8')
    budget = s * t0
    scan = ceil(sum(len(j.activation) for j in jobs) / checks_per_cycle)
    metadata = len(jobs) * metadata_cycles
    # Fail closed before classification if even minimum work cannot fit.
    if scan + metadata > budget:
        return dict(admitted=False, reason='scan/group budget', budget=budget,
                    required_cycles=scan + metadata, tiles=[], jobs=[])
    summaries, groups = [], {}
    for index, j in enumerate(jobs):
        b = weights[j.weight_id]
        keep = tuple(k for k, a in enumerate(j.activation)
                     if abs(a) * max(map(abs, b[k])) > threshold)
        dropped = set(range(len(b))) - set(keep)
        bound = [sum(abs(j.activation[k] * b[k][c]) for k in dropped)
                 for c in range(len(b[0]))]
        summaries.append(dict(batch=j.batch, task=j.task, kept_k=len(keep),
                              dropped_k=len(dropped), error_bound=bound))
        groups.setdefault((j.batch, j.weight_id), []).append((index, keep))
    tiles = []
    for (_, weight_id), rows in groups.items():
        b = weights[weight_id]
        # Stable, bounded-size row packing. No claimed optimal mask clustering.
        for offset in range(0, len(rows), 8):
            chunk = rows[offset:offset + 8]
            union = sorted(set().union(*(set(mask) for _, mask in chunk)))
            for begin in range(0, len(union), k0):
                indices = union[begin:begin + k0]
                # All physical fragments have K0 steps; unused K and lanes zero.
                aa = [[0] * k0 for _ in range(8)]
                bb = [[0] * 8 for _ in range(k0)]
                for kslot, k in enumerate(indices):
                    bb[kslot][:len(b[k])] = b[k]
                    for lane, (idx, mask) in enumerate(chunk):
                        if k in mask:
                            aa[lane][kslot] = jobs[idx].activation[k]
                tiles.append(dict(weight_id=weight_id, rows=[idx for idx, _ in chunk],
                                  k_indices=indices, a=aa, b=bb))
    packed_bytes = len(tiles) * k0 * (8 + 4)
    packing = ceil(packed_bytes / pack_bytes_per_cycle)
    required = scan + metadata + packing
    admitted = required <= budget
    # Fragment accumulation belongs to execution, not the pre-wave planner.
    merge_cycles = sum(len(t['rows']) * len(weights[t['weight_id']][0]) for t in tiles)
    baseline_slots = sum(ceil(len(j.activation) / k0) for j in jobs)
    # Dense fused control isolates fusion gains from sparse-K compaction gains.
    dense_fused_slots = sum(ceil(len(rows) / 8) * ceil(len(weights[key[1]]) / k0)
                            for key, rows in groups.items())
    return dict(admitted=admitted, reason='ok' if admitted else 'packing budget',
                budget=budget, required_cycles=required, scan_cycles=scan,
                metadata_cycles=metadata, packing_cycles=packing,
                schedule_slots=ceil(required / t0), packed_bytes=packed_bytes,
                array_slots=len(tiles), array_cycles=len(tiles) * t0,
                merge_cycles=merge_cycles,
                estimated_total_cycles=budget + len(tiles) * t0 + merge_cycles,
                unfused_dense_slots=baseline_slots, dense_fused_slots=dense_fused_slots,
                jobs=summaries, tiles=tiles if admitted else [])


def execute(plan_result, jobs, weights):
    """Golden execution and tag scatter; Python integer sums, no INT32 overflow."""
    if not plan_result['admitted']:
        raise ValueError('not admitted; execute the pre-existing dense fallback')
    result = {(j.batch, j.task): [0] * len(weights[j.weight_id][0]) for j in jobs}
    for tile in plan_result['tiles']:
        for lane, idx in enumerate(tile['rows']):
            j = jobs[idx]
            for c in range(len(result[(j.batch, j.task)])):
                result[(j.batch, j.task)][c] += sum(a * b[c] for a, b in zip(tile['a'][lane], tile['b']))
    return result
