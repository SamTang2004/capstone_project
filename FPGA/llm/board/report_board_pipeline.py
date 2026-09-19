#!/usr/bin/env python3
"""Compare matched physical board workloads before and after 200 MHz integration."""
import json
import statistics
from pathlib import Path

research = Path(__file__).resolve().parents[1] / 'research'
old = json.loads((research / 'board100/board_results_100.json').read_text())
new = json.loads((research / 'board200/board_results_200.json').read_text())
control = json.loads((research / 'board200/board_results_pipeline100.json').read_text())
fields = ('k', 'rows', 'nominal_density_per_mille', 'shift', 'skip', 'reuse')
counters = ('cycles', 'preparation_cycles', 'activation_requests', 'weight_requests', 'scan_cycles')
def key(c):
    return tuple(c[f] for f in fields)
def indexed(record):
    result = {}
    for c in record['cases']:
        assert c['passed'] and c['results'] == c['expected']
        if key(c) in result:
            assert all(c[f] == result[key(c)][f] for f in counters)
        result[key(c)] = c
    return result
a, b, c = indexed(old), indexed(new), indexed(control)
assert a.keys() == b.keys() == c.keys()
assert new['clock_mhz'] == 200 and control['clock_mhz'] == 100
assert new['bit_sha256'] == control['bit_sha256']
for k in b:
    assert b[k]['results'] == a[k]['results'] == c[k]['results']
    assert all(b[k][f] == c[k][f] for f in counters)
lines = [
    '# Complete sparse overlay at 200 MHz', '',
    f"Physical PYNQ-Z2 validation: {len(new['cases'])} commands at 200 MHz and",
    f"{len(control['cases'])} commands at 100 MHz using the same new bitstream.",
    'Every INT32 result, metadata map and activation request count passes.',
    'Repeated counters are identical, and the same image has identical cycle',
    'counts at both clocks. The host AXI bus remains at 100 MHz.', '',
    '## Timing and area', '',
    'Complete PS/PL implementation on xc7z020clg400-1, Vivado 2024.2:',
    '+0.004 ns setup / +0.024 ns hold slack at 200 MHz, with 0.300 ns user',
    'setup uncertainty on both clocks. No unconstrained internal endpoints or',
    'routing errors. CDC reports ten synchronized crossings and no warnings;',
    'all five bus-skew constraints pass (minimum slack 13.682 ns).', '',
    '| Resource | Original 100 MHz overlay | New 200 MHz overlay |',
    '| --- | ---: | ---: |',
    '| LUTs, including LUTRAM/SRL | 4,204 | 4,323 |',
    '| Fabric flip-flops | 6,564 | 7,774 |',
    '| DSP48E1 | 8 | 8 |',
    '| BRAM tiles | 67.5 | 67.5 |', '',
    'This costs 2.8% more LUTs and 18.4% more flip-flops. The final DRC report',
    'retains two LUT-pair warnings and eight unused-DSP-port power advisories;',
    'bitstream generation passes. The preset DDR-DQS warnings remain historical',
    'board-preset caveats, not a DDR throughput qualification.', '',
    '## Pipeline changes', '',
    '- Registered weight requests and memory output before byte selection.',
    '- Fixed-latency weight reads with a 16-entry response queue and at most 12',
    '  outstanding requests. Credits remove combinational backpressure from BRAM enables.',
    '- Registered activation addresses, memory output, and aligned masks/last markers.',
    '- Local registered upload controls at each weight bank; simplified cache read enable.',
    '- Registered cancellation/reset distribution.',
    '- Board-local AXI-Lite control captures AW/W before decoding, then splits',
    '  register reads into bank selection and response stages.',
    '- AXI3-to-Lite conversion runs at 100 MHz, followed by the generated AXI clock',
    '  converter and the 200 MHz accelerator. Shared DeiT/GEMM RTL is unchanged.', '',
    'Simulation passes 25 memory-harness commands, cancellation at all four weight',
    'pipeline positions, and sustained core backpressure. AXI tests pass 33 writes',
    'and 51 reads with independent AW/W arrivals, byte strobes, response stalls',
    'and reset during a partial transaction.', '',
    '## Measured effect: K=4096, 64 rows, skip + reuse', '',
    '| Nominal density | Old consumer cycles | New consumer cycles | Old preparation + compute (us, 100 MHz) | New preparation + compute (us, 200 MHz) | Speedup |',
    '| --- | ---: | ---: | ---: | ---: | ---: |',
]
for density in (1000, 500, 100, 10, 1, 0):
    k=(4096,64,density,0,1,1)
    before, after = a[k], b[k]
    t0=(before['cycles']+before['preparation_cycles'])/old['clock_mhz']
    t1=(after['cycles']+after['preparation_cycles'])/new['clock_mhz']
    lines.append(f"| {density/10:g}% | {before['cycles']} | {after['cycles']} | {t0:.3f} | {t1:.3f} | {t0/t1:.3f}x |")
lines += ['', 'Preparation cycles increase from 517 to 519 for K=4096. Added latency',
    'includes three extra weight-path drain cycles per eight-row tile and two',
    'extra activation-read stages. Frame latency can overlap useful weight work',
    'at higher densities. Pipelining preserves a sustained',
    'one-column-per-cycle weight path but is not free for short sparse workloads.', '',
    '## Scope and provenance', '',
    'Operands reside in uploaded BRAM. The table includes metadata preparation',
    'and consumer execution; it excludes CPU transfers. It is not a DDR, complete',
    'LLM inference, or fixed-external-bandwidth speedup.', '',
    f"Initial weight upload at the 200 MHz run: {new['weight_upload_seconds']:.3f} s.",
    f"Median host workflow for K=4096 with weights already resident: {statistics.median(x['workflow_seconds'] for x in new['cases'] if x['k']==4096)*1000:.3f} ms.",
    'Python/AXI-Lite upload cost dominates the current host workflow.', '',
    f"BIT SHA256: `{new['bit_sha256']}`.",
    f"HWH SHA256: `{new['hwh_sha256']}`.",
    f"Release host timestamp: `{new['release_host_utc']}`.", '',
    'Raw results and matched timing/resource/CDC reports are in `board200/`.',
    'The original 100 MHz measurements remain in `board100/`.', '']
out = research / 'SPARSE_BOARD_200_RESULTS.md'
out.write_text('\n'.join(lines))
print(out)
