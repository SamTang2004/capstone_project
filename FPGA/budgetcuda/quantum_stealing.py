"""T0-quantized, two-worker steal decisions for an independent ready wave.

This is a snapshot cost model, not a cycle-accurate cache or DMA simulator.
Only queued, unstarted tasks can move; all costs are residual from the snapshot.
"""
from dataclasses import dataclass, replace
from .work_stealing_model import TaskTag


def slots(cycles: int, t0: int) -> int:
    if cycles < 0 or t0 <= 0:
        raise ValueError("cycles must be nonnegative and T0 positive")
    return (cycles + t0 - 1) // t0 * t0


@dataclass(frozen=True)
class QuantumWork:
    uid: str
    tag: TaskTag
    compute_cycles: int
    weight_bytes: int = 0
    activation_bytes: int = 0
    result_bytes: int = 32
    descriptor_bytes: int = 16
    local_memory_cycles: int = 0
    remote_memory_cycles: int = 0
    weight_resident: bool = False
    activation_resident: bool = False
    merge_cycles: int = 0

    def __post_init__(self):
        for name in ("compute_cycles", "weight_bytes", "activation_bytes",
                     "result_bytes", "descriptor_bytes", "local_memory_cycles",
                     "remote_memory_cycles", "merge_cycles"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.compute_cycles == 0:
            raise ValueError("zero-work tasks should be eliminated before scheduling")


@dataclass(frozen=True)
class QuantumMachine:
    t0: int = 16
    bytes_per_cycle: int = 16
    dispatch_cycles: int = 1
    handshake_cycles: int = 4
    decision_cycles: int = 1
    margin_cycles: int = 0

    def __post_init__(self):
        if self.t0 <= 0 or self.bytes_per_cycle <= 0:
            raise ValueError("T0 and bandwidth must be positive")
        if min(self.dispatch_cycles, self.handshake_cycles,
               self.decision_cycles, self.margin_cycles) < 0:
            raise ValueError("costs must be nonnegative")


def local_cost(work: QuantumWork, machine: QuantumMachine) -> int:
    return slots(work.compute_cycles + work.local_memory_cycles
                 + machine.dispatch_cycles + work.merge_cycles, machine.t0)


def evaluate_steals(queue, machine=QuantumMachine(), *, victim_busy=0,
                    thief_ready=0, link_ready=0):
    """Compare every single-task move with no move, preserving remaining order.

    Victim and thief have independent memory service. Payload and result use a
    dedicated serialized link; no other traffic arrives during this snapshot.
    Remote memory latency includes supplied bank stalls. A decision blocks both
    issue paths for decision_cycles. Result return/merge completes before output
    is declared ready. Resident operands are assumed pinned for this decision.
    """
    if min(victim_busy, thief_ready, link_ready) < 0:
        raise ValueError("ready times must be nonnegative")
    if len({w.uid for w in queue}) != len(queue):
        raise ValueError("task IDs must be unique")
    costs = [local_cost(w, machine) for w in queue]
    baseline = max(victim_busy + sum(costs), thief_ready)
    rows = []
    for work, cost in zip(queue, costs):
        payload = work.descriptor_bytes
        payload += 0 if work.weight_resident else work.weight_bytes
        payload += 0 if work.activation_resident else work.activation_bytes
        transfer = (payload + machine.bytes_per_cycle - 1) // machine.bytes_per_cycle
        result = (work.result_bytes + machine.bytes_per_cycle - 1) // machine.bytes_per_cycle
        inbound = slots(transfer + machine.handshake_cycles, machine.t0)
        execute = slots(work.compute_cycles + work.remote_memory_cycles
                        + machine.dispatch_cycles, machine.t0)
        outbound = slots(result + work.merge_cycles, machine.t0)
        decision = slots(machine.decision_cycles, machine.t0)
        remote_finish = max(thief_ready, link_ready, decision) + inbound + execute + outbound
        victim_finish = max(victim_busy, decision) + sum(costs) - cost
        makespan = max(remote_finish, victim_finish)
        rows.append(dict(uid=work.uid, tag=work.tag.__dict__,
                         local_service_cycles=cost, transfer_bytes=payload + work.result_bytes,
                         inbound_cycles=inbound, execution_cycles=execute,
                         outbound_cycles=outbound, decision_cycles=decision,
                         remote_finish=remote_finish, victim_finish=victim_finish,
                         baseline_cycles=baseline, makespan_cycles=makespan,
                         saved_cycles=baseline - makespan,
                         worth_stealing=baseline - makespan > machine.margin_cycles))
    best = min(rows, key=lambda row: row['makespan_cycles'], default=None)
    return dict(baseline_cycles=baseline, candidates=rows,
                selected=best['uid'] if best and best['worth_stealing'] else None)


def split_work(work: QuantumWork, max_compute_cycles: int, *, merge_cycles: int):
    """Proportional K partition for experiments; tail bytes use integer boundaries.

    Every child needs its own descriptor/result and merge. This deliberately
    does not pretend fixed per-task costs shrink with its arithmetic.
    Memory latency is distributed proportionally; real bank traces must replace
    that estimate. UID records the interval in the parent's compute stream.
    """
    if max_compute_cycles <= 0 or merge_cycles < 0:
        raise ValueError("invalid split parameters")
    parts = []
    total = work.compute_cycles
    for start in range(0, total, max_compute_cycles):
        end = min(start + max_compute_cycles, total)
        def portion(value):
            return value * end // total - value * start // total
        parts.append(replace(work, uid=f'{work.uid}[{start}:{end}]',
                             compute_cycles=end-start,
                             weight_bytes=portion(work.weight_bytes),
                             activation_bytes=portion(work.activation_bytes),
                             local_memory_cycles=portion(work.local_memory_cycles),
                             remote_memory_cycles=portion(work.remote_memory_cycles),
                             merge_cycles=work.merge_cycles + merge_cycles))
    return tuple(parts)
