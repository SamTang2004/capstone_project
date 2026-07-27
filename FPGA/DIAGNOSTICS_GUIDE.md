# BudgetCUDA Diagnostics v2

Diagnostics v2 separates two questions that the original report conflated:

1. Why did a layer take so long?
2. At which layer did its numerical result first become wrong?

## Running a comparison

With the v2 PL bitstream and PS firmware loaded:

```powershell
.\.venv\Scripts\python.exe FPGA\tests\hardware\pynq_cnn_diagnostics.py `
  --address 192.168.2.99 `
  --export FPGA\tests\hardware\results\cnn_diagnostics_v2.csv
```

The script runs the same lowered package and seeded input through the board and
the deterministic host mock. It reports the first layer whose raw output hash
differs. `--no-reference` disables this comparison.

## Reading latency evidence

- `compute_duty_percent` is measured systolic-active cycles divided by command
  cycles. This is the useful utilization number.
- `utilization_percent` is retained as a theoretical operations/peak roofline
  estimate. It is not a measured duty cycle.
- High `loader_starvation_percent` means compute requested a tile while the
  look-ahead cache had none ready.
- High `ddr_stall_percent` means the PL memory interface was blocked by
  command, response, or write backpressure.
- High `output_stall_percent` means result production was blocked by the
  writer.
- High cache-full cycles with low starvation means prefetch is sufficiently
  aggressive and more cache depth is unlikely to improve that layer.

The CSV also records DDR commands and beats, descriptor count, core rows, and
cache-ready cycles. These distinguish low useful traffic from a slow response
path.

## Isolating the TileLoader

Run the self-checking TileLoader benchmark without Conv2D, GEMM, the writer,
PS firmware, or the board:

```powershell
& 'C:\Xilinx\Vivado\2024.2\bin\vivado.bat' -mode batch `
  -source FPGA\hardware\vivado\run_lookahead_tile_loader_efficiency.tcl
```

The test submits 32 tiles under three repeatable traffic profiles:

- `direct_ideal`: continuous 32-bit requests with two-cycle DDR responses;
- `direct_constrained`: 12-cycle responses and deterministic 1-in-8 request
  and response stalls;
- `external_ideal`: continuous externally reordered vector fills.

Each result is emitted as one `TILELOADER_EFFICIENCY` line. The important
fields are sustained payload throughput, DDR beat efficiency, output-row
efficiency, first-tile latency, maximum tile gap, stall counts, and maximum
ready-tile occupancy. The test also checks descriptor order, row order, masks,
tile completion, and clean draining.

Current 100 MHz baseline:

| Profile | Payload | DDR efficiency | Output-row efficiency |
| --- | ---: | ---: | ---: |
| `direct_ideal` | 377 MB/s | 95.88% | 32.69% |
| `direct_constrained` | 332 MB/s | 84.40% | 29.12% |
| `external_ideal` | 364 MB/s | N/A | 31.45% |

One tile consumes twenty-four 32-bit words and produces eight 96-bit rows.
Consequently, a single 32-bit fill path can approach 400 MB/s but cannot
sustain a 96-bit output every cycle. The present baseline's maximum ready
occupancy is one tile, confirming that the consumer drains each tile faster
than the fill path can prepare the next one.

## Reading accuracy evidence

Every logical layer records:

- a 64-bit hash of its raw output bytes;
- signed sum, minimum, and maximum;
- logical element, zero, and saturation counts.

The first hash mismatch localizes the failing operator. A sharp increase in
`saturation_percent` points toward bad quantization scale, zero point, or
requantization. Matching ranges with a different hash more often indicates
layout, padding, ordering, or address generation. A mismatch only after
pooling or reshape usually points to shape/layout semantics rather than MAC
arithmetic.

Fused Conv/Requantize/Activation chains do not materialize their intermediate
tensors. Their performance counters are attached to the Conv record and the
fingerprint is attached to the final Activation record.

## Compatibility

The v2 record is 232 bytes per layer and uses diagnostics version `2`. Host,
PS firmware, and the bitstream must be upgraded together. The register bank is
read-only at offsets `0x40` through `0x7c`; `DIAG_MAGIC` must read `DIA2`.
