# LLM bring-up on the existing DeiT overlay

**Superseded direction:** the user has chosen a new independent LLM overlay.
Old-overlay diagnostics below are historical, not a development prerequisite.
See `FPGA/llm/README.md` for current work and board service state.

## Boundary

Keep the current DeiT/HeatViT BIT/HWH and opcode meanings unchanged. This
overlay is initially an integer matrix-engine test platform, not an LLM
decoder. Do not use DeiT LayerNorm in place of RMSNorm, GELU in place of
SiLU-gating, or the patch-embedding opcode for vocabulary lookup.

The existing host QAT module goldens are the LLM numerical reference. A
successful integer GEMM smoke alone does not establish QAT scale, rounding,
requantization, cache, or end-to-end logit equivalence.

## Current board evidence (host UTC 2026-09-11)

- UART/SSH reachable at 192.168.2.99; existing services active.
- Installed BIT/HWH SHA256 match `FPGA/hardware/bcuda_pynq_z2.bit/.hwh`.
  These hashes identify artifacts, not proof that dirty RTL matches the build.
- Original DeiT graph smoke fails at its first LayerNorm, dimensions 3 x 4.
- With daemon stopped, read-only PL dump showed status `0x4`: error latched,
  busy clear, memory error clear under the current top-level status mapping.
  Gamma/beta/input traffic counters reported 20 delivered bytes and three
  misses. This does not identify the values consumed inside the engine.
- The isolated LayerNorm probe confirms exact host readback of input, gamma,
  and beta; it still returns firmware status 7. Its hardware status diagnostic
  field uses the debug-snapshot encoding, not the PL status register encoding.
- Following a clean reload of the same overlay, isolated W8 GEMM (M=3,N=3,K=2)
  returns firmware status 0 and one completed operation, but fails its golden:

  expected: `[15, 7, 4, 37, 11, 6, 59, 15, 8]`

  actual: `[3, 7, 11, 0, 0, 0, 0, 0, 0]`

  Both packed input and packed weight readbacks pass. This is not evidence of
  a missing LLM operator; basic data-path correctness remains unresolved.
- W8 hardware last-op counter was 116 cycles. Do **not** accept this as T0:
  the result was numerically incorrect. The larger diagnostic window includes
  firmware/launch observation time and is not pure array execution latency.

Firmware `bcuda_axi_dma_poll_done` maps both PL error and exhausted polling to
status 7. The captured PL status narrows this occurrence to a latched error,
but not its origin. Readback through the CPU/service does not by itself prove
DDR visibility to PL or correct AXI lane interpretation. Inspect physical
operand visibility, HP0 configuration, packing, and output addressing next.
Do not patch LayerNorm epsilon merely to hide the symptom.

## Reusable and LLM-specific paths

| Component | Existing overlay boundary | LLM action |
| --- | --- | --- |
| Integer matrix engine | 8x8 W8/W4 MAC, INT32 accumulation, tiled operands | First restore exact nonzero GEMM goldens |
| W8 reductions | Current scheduler MAX_K=1024 | Model reductions 288/768 fit statically; board verification required |
| W4 launches | `(K<=192 and N<=768) or (K<=768 and N<=192)` | Direct full 288x288/768/32000 projections do not fit |
| W4 adaptation | Whole-K with smaller N is allowed by source launch guard | Candidate: split output channels into N<=192 slabs; repack B and assemble C with correct strides; no partial-K rounding |
| Normalization/MLP | DeiT LayerNorm and GELU | Retain host/ARM RMSNorm and SiLU-multiply initially |
| Attention | DeiT attention, Softmax length<=256 | Add position-aware RoPE, causal/prefix masking and per-batch persistent KV state separately |
| Embeddings | DeiT patch/class/position embedding | Use the actual vocabulary lookup contract |
| Scheduling | Existing dense execution | Tags/work stealing remain a simulator/host layer until numerical and timing gates pass |

N-splitting is a proposed adapter, not a board-validated workaround. It can
avoid an immediate W4 RTL redesign for K<=768, but increases launches and
requires output layout/scale tests. K>768 or other unsupported shapes must
not silently enter this path. Do not infer that the full QAT model runs merely
because individual W4 slabs satisfy the launch guard.

## Gates before scheduler measurements

1. Pass transport readbacks, then exact nonzero W8 and W4 isolated GEMM.
   Use single operations, not a broad timing sweep. Capture failure before
   reloading; stop daemon before direct MMIO or overlay programming.
2. Compare direct board runtime with the service using the same operands and
   layout. Keep allocations alive while checking physical memory; do not
   inspect freed buffers as if they were stable evidence.
3. Pass M=1 versus padded M=8, K=16/32/64, and N-tail tests. Then test the
   model-shaped whole-K/N-slab adapter against module INT32 goldens.
4. Only for correct outputs, measure array last-op latency separately from
   launch, transfers, packing and completion. Verify clock and count one 8x8
   array (128 arithmetic operations/cycle), not the old two-array default.
5. Integrate existing host LLM modules around the verified GEMM backend;
   compare per-module outputs, KV state, and logits before scheduling research.

## Reproducible probe

`testing/tiny_stories/probe_overlay_contract.py` defaults to readback only;
`--execute` launches one selected operation. It saves JSON even on failure,
including packed operands, expected/actual bytes and diagnostics. It never
programs the FPGA or automatically retries/restarts services.

```powershell
.\.venv\Scripts\python.exe testing/tiny_stories/probe_overlay_contract.py --case gemm-w8 --execute --output testing/tiny_stories/results/overlay_probe_gemm_w8.json
```

Available cases: `layernorm`, `gemm-w8`, `gemm-w4`. Five host tests validate the
probe goldens against the mock backend, including three LayerNorm epsilon
values. Existing transformer host tests also pass (22 tests). This is host
verification only; the board gate remains failed.
