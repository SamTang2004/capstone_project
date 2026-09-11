# Independent LLM overlay — milestone 0

This is a new design root on `LLM_research`, not the DeiT graph overlay with a
different host. Shared reuse is currently limited to the 8x8 integer MAC tile
and its W4 sign-extension logic. Existing dirty shared RTL is not overwritten.

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

1. New versioned AXI-Lite identity/capability and command/completion registers;
   new host driver must reject the old DeiT identity before any launch.
2. Explicit local operand/result storage and transport adapter. Start with
   deterministic PS-driven transfers, then integrate HP0 DDR traffic and time
   it separately. Do not inherit old address/stride assumptions silently.
3. Separate PYNQ block design/build output `llm_pynq_z2.bit/.hwh`, timing and
   resource reports; never overwrite the DeiT artifacts as a build side effect.
4. Exact board goldens for GEMV/GEMM and model-sized tiling. Then add a tagged
   queue/scheduler, communications and T0 characterization.
5. LLM-specific operations and state (RMSNorm, RoPE, causal attention, KV cache,
   SiLU gate) remain host/ARM initially and must match existing QAT goldens.

Milestone 0 is **not** a deployable bitstream, work-stealing scheduler, or full
LLM hardware implementation. The old overlay mismatch investigation is parked,
not resolved, and is no longer a prerequisite for building this new design.
