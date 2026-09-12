# Does a common work quantum improve occupancy?

## Initial answer

A common unit helps compare costs; it does not create parallelism or guarantee
better decisions. Separate (1) load redistribution, (2) locality-aware admission,
(3) cost quantization, and (4) forced reservation padding. A coarse T0 can lose
information; padding can consume the very capacity we want to recover.

The current FPGA has two GEMM cores sharing a scheduler and operand machinery,
not two independently dispatchable workers. These results do NOT describe the
current bitstream's speedup. No FPGA measurements or TinyStories accuracy claims
are made here. No numerical calculations are dropped or split in this model.

## Reproduce

From repository root:

```powershell
.\.venv\Scripts\python.exe -m pytest FPGA/llm/research/test_stealing_experiment.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe FPGA/llm/research/stealing_experiment.py --output FPGA/llm/build/research/stealing_sweep.json
```

23 tests passed. Sweep: six deterministic synthetic workloads, 492 configurations
including controls. Per-task compute and tags are conserved; worker intervals
cannot overlap. Tests check cold-work rejection, probe charging, padding, and
the distinction between temporal and useful-lane utilization. JSON includes
speedup against both unpadded FIFO and FIFO with matching padding/T0 settings.

## Results (synthetic cycles, probe=1 cycle per candidate)

All rows below have identical tasks and resources, with no duration padding.
Cost-aware scanning evaluates queued tasks; simple stealing takes the peer's
tail. T0=1 is the exact-cost control. T0=64 only rounds cost estimates.

| Workload | FIFO cycles | Simple steal | Aware T0=1 | Aware T0=64 |
| --- | ---: | ---: | ---: | ---: |
| Balanced | 780 | 780 | 780 | 780 |
| Skew, cheap migration | 1300 | 852 | 845 | 845 |
| Skew, expensive migration | 1300 | 2374 | 1300 | 1300 |
| Mixed sizes/locality | 1180 | 2488 | 760 | 784 |
| Tiny work | 300 | 236 | 285 | 285 |

Cheap-migration skew: useful compute occupancy rises from 59.1% to 90.9%.
Mixed workloads favor cost-aware selection, but coarse T0 worsens 760 to 784;
no evidence here that quantization itself improves scheduling. Tiny work favors
cheap decisions: evaluating everything costs more than taking one candidate.
Costs of failed probes are included, although they can overlap victim compute.

Forced T0=64 padding changes cheap-migration skew to 1792 cycles, worse than
the original 1300-cycle FIFO. A 64-cycle compute plus one dispatch cycle rounds
to 128 cycles in this deliberately strict whole-service-slot experiment. This
does not imply every useful quantum definition has that penalty: compute-only
slots and separately provisioned dispatch must be tested separately.

The GEMV workload uses 8 of 64 modeled lanes. Work stealing increases temporal
compute occupancy to 90.9%, but useful-lane occupancy is only 11.4%. It cannot
turn an M=1 operation into a full GEMM. Compatible A rows sharing B, K, numeric
format and quantization/layout contract must be batched to recover those lanes;
batch/task tags must follow each output row. Arbitrary different-weight GEMVs
cannot simply be placed in the same ordinary GEMM.

## Model boundaries

Repeated queued-only steals across two hypothetical independent workers. One
serialized admission controller, one serialized inbound migration link. All
tasks ready at time zero; operands are pinned at owners. No eviction, shared
DDR contention, graph barriers, FIFO capacity, uncertainty, dynamic arrivals,
numerical sums, sparse classification or partial-sum merging. Result write/local
memory cost belongs in each task's private wait term; result return is not
modeled. Current fixtures use zero private wait, so they do not validate cache
hit/miss waiting-time prediction. Synthetic migration costs proxy residency.

An accepted candidate is logically reserved at the probe snapshot; only thief
issue waits for the decision. Rejected probes are retried at peer completions,
not each cycle. A physical queue needs reservation/version checking to avoid
selection races. The model is not a cycle-accurate implementation. Exact cost
inputs favor informed admission; predictor error needs a separate sweep. T0
does not reduce the modeled scan cost automatically. Unchanged outcomes with
quantization are not evidence of a cheaper hardware scheduler.

Earlier single-steal model remains in checkpoint d8c6abe under
`FPGA/budgetcuda/quantum_stealing.py`. This experiment extends the question to
repeated decisions and charged rejected probes, not that model's source API.

## Next hardware experiment

1. Measure resident 8x8xK_ref compute start-to-result on the retained systolic
   core. Define T0 from this fixed shape, W4A8 format, alignment and clock.
   Do not use the AXI adapter's replay-to-clear counter as compute T0.
2. Record a cost vector per ready output tile: compute slots, destination-
   specific operand bytes/residency, wait estimate, output credit, live lanes,
   batch/task/output address and confidence margin. Use T0 as a unit, not an
   enforced execution-time promise. Wait and communication remain stochastic.
3. First expose independent issue/completion and operand ownership for the two
   cores. Keep shared-B row batching as a separate control; do not assume the
   current shared scheduler permits arbitrary cross-batch concurrent tasks.
4. Compare static assignment, cheap idle-steal, bounded-candidate cost-aware
   steal, then T0-bucketed admission at the SAME arithmetic and SRAM/bandwidth.
   Charge checks, failed probes, cold weights, contention and metadata traffic.
5. Sweep candidate window, cost error and T0; measure makespan/p95 completion,
   useful MAC lane-cycles, DMA bytes, queue wait, scheduler cycles, LUT and Fmax.
   Only a measured reduction in decision hardware/time that outweighs prediction
   loss supports a benefit from standardizing the quantum itself.

Whole output tiles are the first migration unit. K splitting needs explicit
partial-sum ownership, INT32 overflow semantics and merge-before-requantization
goldens; do not infer numerical equivalence from task-count conservation.
