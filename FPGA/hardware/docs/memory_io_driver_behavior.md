# Memory IO Driver Behavior

This segment defines the PL-side memory IO driver used by the real PS-PL execution path.

## Role

The PS owns allocation and passes physical DDR byte addresses to the PL through control registers. The PL memory IO driver owns deterministic memory transactions once a GEMM/tile operation is started.

The driver sits between tile engines and a single DDR-facing memory command port:

- `tile_loader_ddr` style clients request INT8 reads by byte address.
- `tile_writer_ddr` style clients request INT32 writes by byte address.
- The driver converts those requests into aligned 32-bit memory-port commands.

## Addressing

All addresses at this boundary are byte addresses.

- INT8 read request address `A` reads byte `A`.
- The memory command address for that read is `A & ~3`.
- Returned byte lane is selected from the little-endian 32-bit read word.
- INT32 write request address `C` must be 4-byte aligned.
- INT32 write data is written little-endian to bytes `C + 0` through `C + 3`.
- Misaligned INT32 writes raise `error`.

## Handshakes

Read client:

- `read_req_valid/read_req_ready/read_req_addr`
- `read_data_valid/read_data_ready/read_data`
- The driver may accept one read request while no read response is pending.
- `read_data_valid` must remain asserted until `read_data_ready`.

Write client:

- `write_req_valid/write_req_ready/write_req_addr/write_req_data`
- A write request is accepted only when the driver can issue or hold the corresponding memory command.
- No write response is required by the tile writer contract.

Memory command port:

- `mem_cmd_valid/mem_cmd_ready`
- `mem_cmd_write`
- `mem_cmd_addr`
- `mem_cmd_wdata`
- `mem_cmd_wstrb`
- `mem_read_valid/mem_read_ready/mem_read_data`

## Arbitration

When a read and write are both ready to issue in the same cycle, writes win. This preserves completed accumulator output progress and avoids write starvation during drain.

## Error Behavior

`error` is sticky until reset.

The driver raises `error` for:

- Misaligned INT32 writes.
- Out-of-range memory-model accesses in simulation.
- Unexpected memory read data when no read is outstanding.

## Testbench Expectations

The memory IO golden trace checks:

- Unaligned byte reads from all four lanes of aligned words.
- Aligned INT32 writes with little-endian byte placement.
- Simultaneous read/write issue cycles, where write priority is expected.
- Read response backpressure, where `read_data_valid` must hold stable.
- Final DDR byte contents after all writes.
