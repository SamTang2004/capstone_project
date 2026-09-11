# Queued GEMM AXI adapter — ABI 1

`llm_axi_queued_top` wraps an **unchanged** `bcuda_transformer_ddr_top` and
reuses `axi_lite_control` for its host-facing interface. Its external AXI-Lite
and memory-command ports match the original top, allowing later reuse of the
existing HP0 AXI3 bridge. No bridge, cache, prefetch or scheduler source was
modified. This top is not yet substituted into the board block design.

The adapter captures all sixteen 32-bit staging registers atomically into a
four-entry FIFO on the rising edge of CONTROL.START. It keeps one active task
in addition to the FIFO. A replay FSM programs the original backend registers,
launches GEMM, polls completion, and clears backend CONTROL. This conservative
first integration adds AXI replay/polling overhead; it is not a zero-overhead
scheduler or work-stealing implementation. The separate `llm_tagged_tile`
remains a unit-level experiment; this adapter uses the retained DDR scheduler.

## Register contract

| Offset | Meaning |
| --- | --- |
| 0x00 | CONTROL staging: START bit 0; W4 bit 11; all other bits rejected in v1 |
| 0x04 | Live queue/dispatcher status (below) |
| 0x08, 0x0c, 0x10 | A, B, C physical addresses, as on the old backend |
| 0x14..0x28 | Original strides and M/N/K; plain INT32 GEMM only |
| 0x2c | Batch ID, 32 bits (backend tile size replayed as 8) |
| 0x30 | Task ID, 32 bits (backend array count replayed as 1) |
| 0x34..0x3c | Preserved staging fields; fused requantization not enabled in v1 |
| 0x40 | Read-only identity `0x4c4c4d01` |
| 0x44, 0x48 | Completion batch/task ID |
| 0x4c | Completion backend status, or adapter error code |
| 0x50 | Dispatch-to-backend-clear cycles, modulo 2^32; **not T0** |
| 0x80 | Bit 0 toggles completion ACK and clears sticky enqueue rejection |

Status bits: bit 1 dispatcher non-idle, bit 2 enqueue-ready, bit 3 completion
valid, bit 4 sticky full-queue rejection, bits 7:5 queued count, bit 8 faulted.
Other bits read zero. Enqueue-ready reports FIFO space, not backend health;
software must also check faulted. A faulted dispatcher never launches queued
work until reset.

The new host must validate identity first. This is **not** the old DeiT host
register contract, despite reusing its bus implementation. Raw offsets 0x2c
and 0x30 intentionally belong to this distinct LLM register map.

## Producer/consumer rules

Use one software producer/consumer in v1. Write CONTROL=0, then all command
fields, check ready and faulted, and finally write CONTROL=1 (W8) or 0x801
(W4). Do not keep START high across submissions. Subsequent staging writes
cannot alter queued/active commands. Full-queue doorbells are rejected with
sticky status bit 4; AXI writes themselves still receive OKAY from the reused
slave. A successful AXI write is not proof of enqueue acceptance. Never ignore
bit 4 or clear it before recording which submission failed.

Completion tags/status/cycles stay valid until ACK toggles. ACK does not free
user buffers; the host owns their lifetime. Operands, outputs and layouts must
remain valid until completion. Do not mutate shared weight storage while any
queued task may reference it. Read C only after successful completion.

Backend status bit 2 is error; bit 1 with bit 0 clear is successful completion.
Adapter `0x80000001` rejects an unsupported opcode without launching; dispatch
may continue after ACK. `0x80000002` means an internal AXI response error.
Backend error or internal AXI error latches faulted. ACK releases the completion
record but **does not** restart dispatch. Reset cancels all outstanding work;
the host must invalidate outstanding tags and restore memory ownership/state.

No watchdog or cancellation register is implemented yet. A hung backend can
hold dispatch indefinitely; system/PL reset is required for recovery. The
FIFO is a FIFO, not a deque, and cannot yet steal or reorder tasks.

## Validation

`sim/run_integration.ps1` compiles the original DDR/GEMM top, memory driver,
AXI slave and DSP48E1 simulation model plus the new adapter. A byte-addressed
DDR responder injects command/read/write stalls. This covers the memory-command
boundary, **not** the PS or AXI3 bridge electrical/board behavior.

The integration test passes:

- five queued mixed W8/W4 tasks, including W4 M=1 GEMV;
- every INT32 output checked against a nonzero signed golden;
- independent AW/W arrivals, B/R backpressure and stable responses;
- full FIFO rejection without execution of the rejected output buffer;
- immutable queued snapshots/tags and completion held until ACK;
- unsupported-op rejection, backend invalid-K error, fault-blocked dispatch;
- explicit reset clearing queued and fault state.

Tile and standalone FIFO regressions also run through `sim/run.ps1`. All
results are RTL simulation only. Synthesis, timing closure, board build and
physical golden validation remain required before deployment.
