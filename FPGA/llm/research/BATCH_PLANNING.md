# Pre-batch sparse fusion, fixed array-completion quantum

Reference implementation: `batch_planner.py`; tests: `test_batch_planner.py`.
Research suite: 35 tests pass (12 new planner tests, 23 earlier steal tests).
No RTL or FPGA programming changes in this checkpoint.

## Scheduling boundary

Here batch means a ready compute wave: inputs, weight version and destinations
are known before any tile is issued. It cannot mean an entire autoregressive
inference request when sparsity depends on future layer activations/tokens.
Those are unknown until preceding computation finishes. Static weight metadata
can be prepared offline; activation-dependent decisions run before each ready
wave. No streaming within-wave regrouping is implemented.

Define T0 = completion time of a resident, fixed 8x8xK0 W4A8 tile, with fixed
clock, operand supply and result credits. K0 is even to respect current W4
packing alignment. T0=64 cycles and K0=16 in these tests are SYNTHETIC inputs,
not measured timing. Variable K, memory stalls and blocked results do not have
the same completion time. Padding physical K to K0 is explicit in the golden.
The tile must drain before its context is reassigned in this prototype.
This does not assert all systolic architectures are incapable of finer-grained
context interleaving; that would require a different datapath/state contract.

The hardware prepass would have a fixed issue boundary at S*T0. Successful
early planning waits until that boundary. A missed deadline uses an already
available dense plan, never a partial sparse plan. The Python model checks an
analytical budget; it is NOT a wall-clock-bounded implementation of that FSM.

## 1. What can be skipped?

Each job is a GEMV row A against shared B[K,N], N<=8. Offline metadata contains
max_abs(B[k,:]). Online classification keeps k when

    abs(A[k]) * max_abs(B[k,:]) > threshold

At threshold=0, removed contributions are exactly zero. Positive threshold
removes a whole row's K contribution vector only when every individual output
contribution is small. That is NOT a bound on total error by itself. The
reference computes each output's cumulative sum(abs(dropped contribution))
and checks the actual difference against it. Those extra full-product error
diagnostics are golden-only work, excluded from the proposed hardware prepass.
Network accuracy/perplexity is not evaluated. Quantization scale/domain must
match; real-scale thresholds require the QAT scale metadata.

Counts are K-vector counts, not arbitrary scalar-MAC sparsity. Irregular
per-output weight zeros may remain; a standard dense tile cannot skip each
independent PE multiplication just because some operands are zero.

## 2. What can be fused?

Only rows from the same ready batch and identical weight_id/scale/layout domain
are grouped, at most eight rows. Let R_i be row i's retained K set. Use

    U = union(R_i), fragments = ceil(len(U) / K0).

All rows use the SAME K ordering. At k in U absent from R_i, set that row's
activation to zero. Gather the corresponding B rows once per fragment. Pad
the last fragment with zero A/B values. Tags map each physical output lane
back to (batch, task). Accumulate fragment results before any requantization.
All-zero groups produce tagged zero outputs without issuing array work.

Rows are packed in stable input order, not with a claimed optimal clustering
algorithm. Different weights cannot share a normal GEMM merely because they
have complementary holes. What fills idle capacity is retained useful work;
the discarded work itself is not packed into another task.

## Fixed-shape example and full-cost accounting

Eight rows, K=32,N=8; each row retains 4 K positions (87.5% skipped).
Each input activation is 2 at retained positions and zero elsewhere. Tests
use signed W4 weights and verify exact output goldens and tag scatter.

| Model metric | Shared retained positions | Disjoint retained positions |
| --- | ---: | ---: |
| Union K count | 4 | 32 |
| Sparse fused array slots | 1 | 2 |
| Dense fused array slots | 2 | 2 |
| Unfused dense array slots | 16 | 16 |
| Estimated prepass cycles | 72 | 96 |
| Minimum S, T0=64 | 2 | 2 |
| Estimated fragment accumulation cycles | 64 | 128 |
| S*T0 + sparse array + accumulation | 256 | 384 |

The prepass uses 8 K checks/cycle, 2 metadata cycles/job, 8 packed bytes/cycle.
Each fixed fragment writes 16*(8 A bytes + 4 W4 bytes)=192 bytes. Fragment
accumulation is conservatively charged one cycle per active output per fragment.
If dense fused inputs/plans are already prepared, its two-slot execution plus
the same accumulation model costs 256 cycles. Thus this small aligned example
only TIES that control after planning overhead; the disjoint example loses.
The 16-to-1 tile reduction alone would misleadingly credit fusion gains to
sparsity. Dense fused planning/packing and output costs must be measured in a
fair end-to-end comparison; the existing-prepared-plan case is only one control.

The API default S=32 is merely a permissive functional-test budget. For the
example set S=2; using S=32 in a fixed-boundary design would waste 2048 cycles
before execution. The returned estimated_total_cycles intentionally charges
the requested S, not just the minimum measured/modelled planner requirement.
Admission currently means budget-feasible, NOT automatically profitable.

## Budget and steal admission architecture

Proposed prepass stages: bounded metadata scan -> classify -> row grouping/K
union -> descriptor and operand packing -> issue at S*T0. Require explicit
input-resident status, enough scratchpad/queue capacity, and a dense fallback.
The Python implementation uses variable-size sets/lists; no fixed hardware
throughput, storage capacity or Fmax has been established. Metadata precomputation
and shared DDR reads are not included in its budget. It recomputes metadata in
Python for convenience; proposed hardware stores it offline. Rejected candidate
plans are not executable and execute() fails closed.

For a ready fixed tile, reserve whole completion slots on an available worker.
Use the predicted end-to-end wave finish, including source queue wait,
destination operand traffic/residency, descriptor checks and completion/merge,
to accept a migration only if it beats the no-move finish by an uncertainty
margin. A fixed T0 does not remove these other costs. Fusion eligibility and
steal profitability are separate decisions. This planner implements the former
and budget gating; no physical worker assignment or transport admission yet.

Next measured experiment: calibrate fixed-tile T0, bound scan/packing throughput,
and sweep S with identical ready waves under dense-unfused, dense-fused and
sparse-fused controls. Then expose independent worker issue with pinned operand
ownership. The current two cores still share a scheduler; do not equate them
with GPU SMs. Track useful MAC lane-cycles as well as completion counts.

Python goldens use unbounded integer accumulation. Matching actual INT32 wrap,
fragment buffers, per-row requantization and model-scale accuracy are required
before this is deployed or K splitting is claimed bit-exact on hardware.
