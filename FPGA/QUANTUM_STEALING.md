# T0-quantized steal admission experiment

T0 defines a reservation quantum: q(c) = ceil(c / T0) * T0. It does not
guarantee arbitrary DDR operations finish at a fixed time. Deterministic compute
slots require resident operands, a reserved bank/port and an output credit.
Otherwise a task remains waiting, or needs an explicit overrun/replay contract.
Padding costs are real; a one-cycle excess can consume another whole quantum.

For one independent ready wave and one possible queued-task migration:

- F0 = max(victim_busy + sum(local service costs), thief_ready).
- Fv = max(victim_busy, decision cost) + sum(local costs except stolen task).
- Ft = max(thief_ready, link_ready, decision cost) + inbound + execute + return.
- Admit only when F0 - max(Fv, Ft) exceeds a configurable uncertainty margin.

Local service includes compute, dispatch, local memory stall and merge.
Inbound includes descriptor, nonresident weights/activations and handshake.
Return includes the result payload and merge. Each service stage is rounded
up to T0. Operand residency is destination-specific and assumed pinned.

Implementation: `budgetcuda/quantum_stealing.py`.
Reproduce synthetic sweep: `python testing/tiny_stories/run_quantum_stealing.py`.

## Synthetic examples, not measured LLM speedups

At T0=16, bandwidth=16 bytes/cycle, victim busy for another 128 cycles,
the queue holds 128-cycle cold work with 64 KiB weights, 64-cycle work with
resident weights, and 4-cycle tiny work. No stealing finishes at cycle 368.

| Move | Wave completion | Decision |
|---|---:|---|
| Cold large task | 4288 | reject |
| Resident medium task | 288 | choose |
| Tiny task | 352 | helps less |

For a separate 256-compute-cycle task with 16 KiB cold weights, adding two
merge cycles per chunk gives:

| Chunk compute cycles | Tasks | No steal | Best outcome with at most one steal |
|---:|---:|---:|---:|
| 256 | 1 | 400 | 400 (reject) |
| 128 | 2 | 416 | 416 (reject) |
| 64 | 4 | 448 | 384 |
| 32 | 8 | 512 | 464 |
| 16 | 16 | 640 | 608 |

Splitting to 64 improves the unsplit baseline by only 4%, not the 14.3% one
would infer using the already-split baseline. Splitting further is worse.
The local owner is assumed to have its operands; remote transfer is additional.

## Boundaries and next steps

This is a two-worker snapshot admission model with supplied residual wait
times, independent memory service and a dedicated link. It evaluates alternative
single moves, not repeated simultaneous steals. It has no cache replacement,
shared-DDR interference, bank-level event simulation, or numerical accumulation.
The scalar decision cost is a parameter, not a synthesized comparator network.
Rejected-probe overhead and repeated polling are not modeled; F0 is the
no-probe baseline. These must be added before system throughput claims.

K splitting preserves compute count and operand bytes by integer interval
partition, and duplicates descriptor/result costs per fragment. Real address
alignment, active-K indices and partial-sum ownership must replace this estimate.
Use integer partial sums, merging before requantization, to avoid changing the
numerical contract. Do not migrate running tasks without a separate state-save
protocol. Prefer whole output tiles when that avoids partial-sum traffic.

Next: feed real safe-sparsity traces and bank events, compare T0=1/4/8/16/32,
count padding and rejected probes, and compare unsplit/static, idle-only steal,
and cost-aware admission with identical arithmetic and SRAM resources.
