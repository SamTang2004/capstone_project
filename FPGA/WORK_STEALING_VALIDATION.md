# TinyStories sparse work-stealing validation model

## Question

How much scheduler overhead can a sparse W4 decoder tolerate before dynamic
load balancing costs more than it saves?

The validation model uses the real W4 codes from the 500-step TinyStories QAT
checkpoint. A weight is provisionally significant when `abs(code) >= 4`.
This selects 5.44% of individual parameters, but an 8-output GEMM K step is
active when any lane is significant, so active compute is 28.0% of dense K
steps. This threshold is a scheduler experiment, not an accuracy-approved
pruning policy.

## Model

The model preserves all 43 sequential GEMM dependency barriers in one decode
token: seven matrices per transformer layer across six layers, followed by the
vocabulary projection. It does not pool independent work across unavailable
future layers.

One whole-tile task is `(operation, 8-output N tile, complete K reduction)`.
Alternative experiments split K into chunks of 64, 32, or 16 steps. Every
active K step costs one GEMM-core cycle. Scheduler costs are explicit:

- local task dispatch;
- steal probe and transfer;
- partial-result commit;
- operation barrier;
- descriptor bytes.

Static round-robin, a serialized central queue, and distributed work stealing
run over identical work. The pathological ordering deliberately assigns light
tasks to one worker and heavy tasks to the other, representing the motivating
load-1/load-2 failure.

## Results

Recommended comparison: two workers, one-cycle local dispatch, eight-cycle
total steal handshake, 16-byte descriptors, one-cycle commit, two-cycle
operation barrier, and steal batches of at most four tasks.

| K granularity | Scenario | Tasks | Descriptor | Static cycles | Work stealing | Speedup | Utilization |
|---|---|---:|---:|---:|---:|---:|---:|
| whole tile | actual | 6,232 | 97.4 KiB | 275,255 | 274,650 | 1.002x | 96.9% |
| whole tile | pathological | 6,232 | 97.4 KiB | 312,744 | 274,960 | 1.137x | 96.8% |
| K=64 | pathological | 27,786 | 434.2 KiB | 358,582 | 296,590 | 1.209x | 89.7% |
| K=32 | pathological | 44,712 | 698.6 KiB | 360,046 | 313,765 | 1.148x | 84.8% |
| K=16 | pathological | 79,279 | 1,238.7 KiB | 400,655 | 349,741 | 1.146x | 76.1% |

The whole-tile sparse ideal is 266,117 cycles. Work stealing reaches 274,650
cycles, about 3.2% above that dependency-aware ideal. For context, dense
single-core execution is 1,898,496 modeled cycles and ideal dense N-tile split
across two cores is 949,248 cycles. Those ratios measure scheduling potential
only; they do not include sparse-index memory traffic or establish pruning
accuracy.

## Overhead budget

Whole-tile task dispatch must be cheap:

- actual ordering stops beating static when recurring local dispatch exceeds
  approximately one cycle per task;
- pathological ordering tolerates up to 12 cycles per task;
- with local dispatch fixed at one cycle, the conservative continuous steal
  handshake budget is 87 cycles for actual ordering and 698 cycles for the
  pathological ordering.

The actual workload needs only 31 steals under the recommended configuration,
so steal latency is much less dangerous than adding several control cycles to
all 6,232 tasks.

## Initial architectural decision

Implement distributed local queues rather than a serialized central queue:

1. Partition whole N tiles statically between the two cores.
2. Pop one local descriptor per cycle.
3. Continue locally while work remains.
4. Probe the peer only when the local queue is empty.
5. Steal up to four tasks, never more than half the victim's queued work.
6. Keep GEMM-operation barriers explicit.
7. Target at most 32 cycles for a complete steal handshake, leaving margin
   below the 87-cycle conservative actual-workload bound.

Do not split K in the first RTL experiment. K=64 can improve the deliberately
pathological case, but it multiplies descriptor storage by 4.5 and lowers
utilization. K=32/K=16 are already inferior even though the model assumes an
optimistic one-cycle partial-result commit; real accumulator merge traffic
would make them worse.

## Missing effects before an RTL commitment

- Measure sparse-index and irregular weight-fetch DDR traffic.
- Define how an active-K list addresses packed W4 vectors.
- Replace the optimistic one-cycle output commit with the proposed accumulator
  ownership/writeback design.
- Sweep significance thresholds against language-model perplexity; `abs(code)
  >= 4` currently has no accuracy authorization.
- Feed synthesized queue/control timing and BRAM costs back into this model.
- Add backpressure and cache-miss traces from the existing memory driver.

The model and CLI are in `budgetcuda.work_stealing_model` and
`testing/tiny_stories/run_work_stealing_model.py`. The generated JSON retains
all policy samples so later RTL measurements can replace assumptions without
changing the comparison framework.

## Tagged multi-batch extension

The second validation model combines two batches in each of the same 43 GEMM
dependency waves. Every task carries an immutable composite tag:

`(batch_id, operation_id, generation)`

The descriptor still carries output tile and K-range fields. Initial queue
ownership is `batch_id % worker_count`; a stolen task retains its complete tag,
and the completion ledger is built from tasks actually executed by workers.
A multiset check detects lost or duplicated tasks after stealing.

For two batches, 43 operations, and one generation, the minimum tag is eight
bits: one batch bit, six operation bits, and one generation bit. This adds one
byte per descriptor, or 12.2 KiB across 12,464 whole-tile tasks.

At one cycle of tag lookup per task and an eight-cycle steal handshake:

| Batch active-work factors | Makespan | Speedup vs static | Utilization | Steals | Tag errors | Tag-cost break-even |
|---|---:|---:|---:|---:|---:|---:|
| 1.00 / 1.00 | 550,991 | 0.989x | 97.0% | 0 | 0 | none |
| 0.50 / 1.00 | 422,181 | 1.290x | 95.0% | 515 | 0 | 16 cycles/task |
| 0.25 / 1.00 | 356,556 | 1.528x | 93.0% | 710 | 0 | 24 cycles/task |
| 0.05 / 1.00 | 305,165 | 1.785x | 91.9% | 848 | 0 | 39 cycles/task |

The equal-load case correctly performs no steals; its roughly 1.1% loss is
pure tag lookup cost. Therefore tag propagation should be a one-cycle pipeline
stage or ride alongside the descriptor with no separate lookup, and stealing
should remain strictly idle-triggered. The extreme 5%/100% case crosses below
static performance between 39 and 40 tag-lookup cycles per task.

This is deliberately conservative about dependencies: batches share a global
barrier between GEMM operations. It does not yet model independent batch graph
progress, descriptor-generation/enqueue latency, sparse-index DDR traffic, or
completion backpressure. Those effects belong in the next event-driven model.
