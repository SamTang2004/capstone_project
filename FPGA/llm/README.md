# Independent LLM overlay — milestone 0

This is an additive design root on `LLM_research`. The branch was checkpointed
at `d8c6abe` (branch `codex/checkpoint-llm-before-deit-20260911`), then restored
to the exact DeiT `a0f1720` tree by commit `f07f154` without rewriting history.
Only this independent LLM module directory was selectively recovered afterward.
Ignored model/golden/build files remain on disk, not in the Git checkpoint.

Preserve the DeiT AXI-Lite controller, HP0 AXI3 bridge, DDR memory driver,
request pipelines, prefetch/cache machinery and GEMM scheduler. Do not rebuild
the bus or discard bandwidth optimizations. The new queues and tagged dispatch
will be integrated at command/completion boundaries. Existing AXI/DDR
bus sources remain byte-for-byte equal to the DeiT baseline. The shared GEMM
scheduler adds a default-on fused-writer generate parameter; LLM disables it.

## Implemented

`rtl/llm_tagged_tile.sv` is a synthesizable, single-outstanding compute boundary:

- command handshake captures a 16-bit batch ID and 32-bit task ID;
- M/N are 1..8, K is 1..65535; M=1 is GEMV;
- each accepted input contains eight signed A8 lanes and eight B8 lanes, or
  eight signed W4 lanes in the low 32 bits (low nibble is column zero);
- completion returns a full row-major 8x8 INT32 tile, zeroing inactive lanes;
- result, tags, status and counters remain stable until completion acceptance;
- invalid shape returns status 1 without launching; core error returns 2;
- reset cancels in-flight work; host must invalidate outstanding tags on reset;
- execution_cycles counts clock edges after command acceptance through result
  capture; input_wait_cycles counts RUN edges ready for operands but empty;
  completion backpressure is excluded. Counters wrap at 2^32 cycles.

There is no compulsory T0 padding. First measure a fixed successful shape
without stalls; record wait and communication costs separately. There is no
operand-starvation watchdog yet; reset is required to cancel an incomplete tile.

## Simulation

`rtl/llm_command_queue.sv` adds a parameterized FIFO for whole tagged descriptors
(default 128 bits, four entries). Depth must be positive. It supports a push
on a full queue when simultaneously popping, explicit non-power-of-two wrap,
and stable output under backpressure. There is no empty-queue bypass. Reset
cancels queued descriptors. It is not a work-stealing deque.
`llm_axi_queued_top.sv` now connects a 512-bit
instance to the reused AXI-Lite slave and GEMM-only DDR top through an
internal register-replay FSM. See `AXI_QUEUE_ABI.md` for the separate LLM map,
enqueue/ACK/fault rules, tests and current limitations.

Run `FPGA/llm/sim/run.ps1` from PowerShell. It uses Vivado 2024.2 XSim with
the real DSP48E1 simulation model, not a mock accelerator. Build artifacts stay
under `FPGA/llm/build/sim`. Tests cover nonzero signed W8/W4 multiplication,
inactive lanes, batch/task return, input bubbles, completion backpressure,
invalid command rejection and subsequent command recovery.

## Board state

Old `budgetcuda.service` and `budgetcuda-fpga.service` were stopped and disabled
at the user's request. TCP 9000 is closed; SSH/UART remain available. No BIT/HWH
or old server files were deleted. Disabling services removed only their boot
enablement symlinks. Restore with:

```sh
sudo systemctl enable budgetcuda-fpga.service budgetcuda.service
sudo systemctl start budgetcuda.service
```

The old PL configuration is still physically present (`fpga0/state=operating`),
although detached from its stopped daemon. It has **not** been erased or
replaced. Programming a validated new image will replace it. Do not call a
metadata reset or stopped systemd unit a physical bitstream unload.

## Still required before a deployable overlay

1. Add the host driver for the now-implemented versioned AXI queue adapter;
   it must check identity and enqueue acceptance before launch. The AXI slave remains unchanged; the distinct LLM-facing map uses tags.
2. Validate the integrated `llm_axi_queued_top` and existing HP0 AXI3 bridge on
   the board. Queue-to-DDR-command integration now passes RTL simulation; PS/HP0
   and board checks remain. Time replay, compute and transport separately.
3. Separate PYNQ block design/build output `llm_pynq_z2.bit/.hwh`, timing and
   resource reports; never overwrite the DeiT artifacts as a build side effect.
4. Exact board goldens for GEMV/GEMM and model-sized tiling. Then add a tagged
   queue/scheduler, communications and T0 characterization.
5. LLM-specific operations and state (RMSNorm, RoPE, causal attention, KV cache,
   SiLU gate) remain host/ARM initially and must match existing QAT goldens.

Milestone 0 is **not** a deployable bitstream, work-stealing scheduler, or full
LLM hardware implementation. The old overlay mismatch investigation is parked,
not resolved, and is no longer a prerequisite for building this new design.

## GEMM-only pruning (2026-09-11)

`llm_gemm_ddr_top.sv` replaces the multi-operator DeiT backend. LayerNorm,
Softmax, HeatViT pruning, elementwise residual/GELU, standalone requant and
packing engines, fused requant/GELU writer and their operator arbitration are
absent from the LLM elaboration. Their reference sources remain under the
shared hardware directory, excluded by `vivado/rtl_sources.txt`. Both XSim
integration and Vivado consume that explicit manifest. AXI, HP0 bridge, DDR
cache, request skid stages, dual GEMM cores and W4 weight prefetch/cache remain.
The shared scheduler's new parameter defaults to enabled for existing callers;
only LLM sets it to zero. Unsupported operations are rejected, not emulated.

This is a plain INT32-output W4A8/W8A8 GEMM engine, not complete LLM inference.
Non-GEMM LLM operations must remain on the host until separately designed.
