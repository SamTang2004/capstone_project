#!/usr/bin/env python3
"""Summarize physical board ablations without treating CPU uploads as free."""
import argparse
import json
import statistics
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('results', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
record = json.loads(args.results.read_text())
cases = record['cases']
if not cases or not all(c['passed'] and c['results'] == c['expected'] for c in cases):
    raise RuntimeError('Incomplete or failing board results')
groups = {}
for case in cases:
    key = tuple(case[x] for x in ('k', 'rows', 'nominal_density_per_mille', 'shift', 'skip', 'reuse'))
    signature = tuple(case[x] for x in ('cycles', 'preparation_cycles', 'activation_requests', 'weight_requests', 'scan_cycles'))
    groups.setdefault(key, set()).add(signature)
if any(len(v) != 1 for v in groups.values()):
    raise RuntimeError('Hardware counters changed between repeated identical cases')
clock = record['clock_mhz']
lines = [
    '# Sparse engine: physical PYNQ-Z2 results', '',
    f"{len(cases)} commands passed exact signed INT32 golden checks at {clock:g} MHz.",
    'Metadata maps and activation request counts also passed. Hardware cycle counts',
    'are identical across repeated cases.', '',
    f"- Board: `{record['board']}`; ABI `{record['abi']}`.",
    f"- Release host timestamp: `{record['release_host_utc']}`.",
    f"- BIT SHA256: `{record['bit_sha256']}`.",
    f"- HWH SHA256: `{record['hwh_sha256']}`.", '',
    '## K=4096, 64 output rows', '',
    'All modes use the same signed operands and eight DSP lanes. Density is the',
    'nominal generator setting; the survivor column reports actual nonzero values.',
    'Each cell is consumer execution cycles. Preparation is a separate measured',
    'pass required by this harness, including when skipping is disabled.', '',
    '| Density | Survivors | Raw rescan | Skip only | Reuse only | Skip + reuse | Combined + preparation (us) | Raw / combined, including preparation |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
]
for density in (1000, 500, 100, 10, 1, 0):
    selected = [c for c in cases if c['repeat'] == 0 and c['k'] == 4096 and c['nominal_density_per_mille'] == density]
    modes = {(c['skip'], c['reuse']): c for c in selected}
    ordered = [modes[m] for m in ((0,0),(1,0),(0,1),(1,1))]
    raw, combined = ordered[0], ordered[-1]
    total = combined['cycles'] + combined['preparation_cycles']
    ratio = (raw['cycles'] + raw['preparation_cycles']) / total
    values = ' | '.join(str(c['cycles']) for c in ordered)
    lines.append(f"| {density/10:g}% | {raw['survivors']} | {values} | {total/clock:.3f} | {ratio:.3f}x |")
lines += ['', '## Measurement limits', '',
    'Operands are uploaded by CPU through AXI-Lite into 4 KiB activation and 256 KiB',
    'weight BRAM storage. CPU upload and result retrieval are excluded from the',
    'hardware cycle counts. JSON separately records initial weight upload time and',
    'per-command activation upload, command/polling, result retrieval, and workflow',
    'wall times (with weights already resident). These results',
    'are not DDR bandwidth, complete LLM inference, energy, or frontier comparisons.', '',
    f"Initial 256 KiB weight upload: {record['weight_upload_seconds']:.3f} s. For K=4096,",
    f"median activation upload: {statistics.median(c['activation_upload_seconds'] for c in cases if c['k']==4096)*1000:.3f} ms;",
    f"median complete command workflow with resident weights: {statistics.median(c['workflow_seconds'] for c in cases if c['k']==4096)*1000:.3f} ms.",
    'Python/AXI-Lite transfers dominate this host-driven workflow, so the hardware',
    'cycle gains do not imply comparable application wall-time speedups.', '',
    'Raw rescan is the same sparse consumer with skip/reuse disabled, not an',
    'independently optimized dense GEMM. Preparation would be avoidable in a',
    'production no-skip path; charge that distinction in future architecture comparisons.', '',
    'At 0.1% density, combined preparation + consumption is 694 cycles (6.94 us).',
    'Reuse-only consumption takes 825 cycles: if its unnecessary metadata pass is',
    'removed, the combined approach is only 1.19x faster than that stronger baseline.',
    'At 1% density, combined 1,360 cycles is slower than reuse-only consumption',
    '(1,085 cycles). An adaptive choice or metadata generated upstream remains useful.', '',
    '## Complete implementation', '',
    'Vivado 2024.2, xc7z020clg400-1, 100 MHz, 0.300 ns user setup uncertainty:',
    '+1.017 ns setup / +0.036 ns hold slack, no unconstrained internal endpoints,',
    'and no routing errors. Total: 4,204 LUTs (including 60 SRLs), 6,564 fabric FFs,',
    'eight DSPs, 66 RAMB36 + three RAMB18 (67.5 BRAM tiles). Bitstream generation',
    'passes. DRC retains eight LUT-pair warnings and eight unused-DSP-D-port power',
    'advisories; the board preset also emits inherited DDR-DQS delay warnings.', '',
    'The complete 200 MHz integration failed routed setup at -1.947 ns and was not',
    'programmed. The earlier isolated core passed 200 MHz, but this board harness',
    'adds memory and control paths. The worst 200 MHz path crosses from survivor',
    'cache BRAM to weight BRAM addresses; register that boundary before retrying.', '',
    'Reproduce with `board/DEPLOYMENT.md`. Preserve the raw JSON, stdout, routed',
    'reports and release manifest alongside this summary.', '']
args.output.write_text('\n'.join(lines))
print(args.output)
