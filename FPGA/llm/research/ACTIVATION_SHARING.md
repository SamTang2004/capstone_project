# Activation sharing experiment

This experiment asks how much it helps to classify one activation vector once
and reuse its survivors across output rows of `y = W x`. It is an executable
architectural performance model, not new RTL, a synthesis result, or a board test.
The existing eight-context/eight-multiplier core motivates the compute limits;
its detailed handshake timing is not reproduced.

## Compared modes

- **dense_cached**: cache activations once, read every weight, execute every
  product (including zeros). Choose the faster of row-oriented and broadcast
  ordering under the same physical layout. This is a strong dense control.
- **repeated_finder**: classify the same activation separately for every output
  row, then execute the surviving products in row order within eight-row waves.
- **shared_replay**: classify once and store `(index, activation)` descriptors;
  replay them for every output row. Weight requests and compute scheduling are
  identical to repeated_finder, isolating the effect of sharing.
- **shared_broadcast**: classify once, then broadcast each retained activation to
  up to eight output rows. Fetch a different weight for each row. Repeat the
  descriptor sequence for each eight-row wave. This also changes request order.

Repeated finding is an ablation, not a claim about the best existing accelerator.
An implementation that already caches selection metadata already has this reuse.

## Model contract

1. Activations and weights are signed eight-bit integers. Keep an activation if
   `abs(x) >> X` is nonzero; its original value is preserved. Shift zero removes
   only exact zeros. Positive shifts are approximate pruning and require a
   separate model-quality evaluation before any LLM accuracy claim. The dense
   control computes the same pruned mathematical function.
2. Activation memory supplies 8 or 64 bytes/cycle, independently of weight banks.
   Assemble 64-position frames; classification has a fixed six-cycle delay.
   Eight frame slots bound occupancy; an output serializer writes up to eight
   descriptors/cycle. Loading overlaps classification and descriptor drain.
   This models a possible finder interface, not cycle equivalence to the RTL.
3. Preparation completes before weight execution. Repeated mode charges one
   complete pass per row; shared modes charge one per matrix-vector operation.
   The CSV also reports `max(preparation, compute)` as an optimistic perfect
   overlap bound, to expose sensitivity to this phase separation. Neither the
   sum nor the bound is a measured existing-RTL latency.
4. Eight single-read byte-wide weight banks, two-cycle response latency. Address
   is `row*K+k` for row-major or `k*rows+row` for output-interleaved. Layout is
   pre-established; transposition, matrix loading and layout conversion are not
   included. These are alternative physical organizations of the same matrix.
5. Admit up to eight candidate products/cycle into a 64-entry request window.
   Greedily issue at most one per bank and at most eight total/cycle. Responses
   plus outstanding reads occupy a separate 64-entry capacity. The eight-lane
   core consumes up to eight ready products/cycle, with three drain cycles and
   one command cycle per eight-row wave; empty waves add one control cycle.
   Finite buffering and bank conflicts are charged. The scheduler itself has
   no synthesized area/timing estimate and may need further pipeline stages.
6. Sparse modes discard zero weights only after reading them. No oracle gives
   their locations. Accumulation wraps modulo 2^32; rows above eight require
   sequential waves, while shared descriptors remain resident across waves.
7. Descriptor storage includes activation plus byte-rounded index. Replaying
   needs up to eight descriptor reads/cycle; broadcast reuses a descriptor across
   its row group. Logical bytes do not imply free ports: banking, padding, SRAM
   allocation and any replication require an RTL design and physical costing.
8. Source-byte metrics are local memory traffic. They exclude descriptor traffic
   (reported separately), matrix preload, result writes, DDR bursts, DMA and
   external-memory arbitration. There is no energy or end-to-end inference claim.

## Verification and reproduction

From the repository root:

```powershell
python -m unittest discover -s FPGA/llm/research -p test_activation_sharing.py
python FPGA/llm/research/activation_sharing_experiment.py
```

Every sweep case checks all four modes against an independent scalar matrix-vector
sum. Directed tests cover signed threshold boundaries including -128, all shifts,
partial frames, empty vectors, zero weights, W4-range weights, partial output waves,
single-row no-reuse equivalence, traffic invariance, bank-conflict penalties,
finite response backpressure, and INT32 overflow. Three deterministic seeds and
random/clustered/bank-clustered masks expose sensitivity to placement of survivors.

Generated files: `ACTIVATION_SHARING_RESULTS.md` and
`ACTIVATION_SHARING_PERFORMANCE.csv`. These do not modify earlier RTL results.

## Next hardware gate

Implement a descriptor store and replay controller, connect the finder source
indices to it, and supply weights through a real banked-memory interface. First
retain row order to isolate sharing, then add broadcast. Verify against the same
matrix-vector fixtures before synthesis. Compare dense/shared selection using
the same bank layout and resource budget; qualify the complete changed path at
100 MHz rather than inheriting the previous timing result.
