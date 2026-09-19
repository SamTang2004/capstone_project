"""Compare cross-bank merging with the recorded bank-dispatch experiment."""
from pathlib import Path
import csv
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
CASES = ['dense_w8', 'half_w8', 'ten_percent_w8', 'point_one_x_99', 'zero_jobs', 'short_jobs']


def read(path):
    return path.read_text(encoding='utf-8', errors='replace')


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def key(r):
    return r['case'], int(r['mode']), int(r['limited_supply'])


def synth(folder):
    util, timing = read(folder/'utilization.rpt'), read(folder/'timing_synth.rpt')
    counts = [int(re.search(pattern, util)[1]) for pattern in [
        r'\| Slice LUTs\*\s*\|\s*(\d+)', r'\| Slice Registers\s*\|\s*(\d+)',
        r'\| DSPs\s*\|\s*(\d+)', r'\| Block RAM Tile\s*\|\s*(\d+)']]
    assert counts[2] == 8
    slacks = [re.search(pattern, timing)[1] for pattern in [
        r'Slack \([^)]+\)\s*:\s*([-\d.]+)ns\s*\(required time - arrival time\)',
        r'Slack \([^)]+\)\s*:\s*([-\d.]+)ns\s*\(arrival time - required time\)']]
    return counts + slacks


def main():
    unit = re.search(r'PASS: wide finder frames=(\d+) chunks=(\d+) cycles=(\d+) stalls=(\d+) replacements=(\d+)',
                     read(ROOT/'build/merged_finder/xsim.log'))
    end = re.search(r'PASS: wide sparse dot8 real DSP48E1 checked=(\d+) cycles=(\d+)',
                    read(ROOT/'build/merged_sparse_dot/xsim.log'))
    assert unit and end, 'Missing passing merged simulations'
    raw = ROOT/'build/merged_sparse_dot/wide_performance.csv'
    current = rows(raw)
    old = {key(r): r for r in rows(ROOT/'research/WIDE_FINDER_PERFORMANCE.csv')}
    new = {key(r): r for r in current}
    assert new.keys() == old.keys()
    for k, r in new.items():
        for field in ['source_slots', 'useful_products', 'declared_jobs', 'source_packets', 'supply_wait_cycles']:
            assert r[field] == old[k][field], (k, field)
        if k[1] in (0, 4):
            assert r == old[k], ('anchor changed', k)
    # The default parameter must preserve the previous test run exactly.
    assert rows(ROOT/'build/wide_sparse_dot/wide_performance.csv') == list(old.values())
    assert 'PASS: wide sparse dot8' in read(ROOT/'build/wide_sparse_dot/xsim.log')
    assert 'PASS: wide finder' in read(ROOT/'build/wide_finder/xsim.log')
    lines = ['# Cross-bank merger measured results', '',
             'Same eight-DSP arithmetic core, fixtures, input widths and source-supply models as the bank-dispatch baseline.', '',
             f'Standalone merged finder: **{int(unit[1]):,} frames**, {int(unit[2]):,} chunks, {int(unit[3]):,} cycles; '
             f'{int(unit[4]):,} stall cycles and {int(unit[5]):,} simultaneous frame replacements. '
             'Checks include every byte/threshold/position, every survivor count 0–64, full intermediate beats, paired payloads, indices, metadata, and reset after a partially drained frame.',
             f'Real-DSP comparison: **{end[1]} result vectors / {sum(int(r["declared_jobs"]) for r in current)} dot results**, passing in {int(end[2]):,} scoreboard cycles. '
             'Includes W4 thresholds, source/result stalls, INT32 wrap, reset and back-to-back waves. '
             'Default bank-dispatch regression and both scan-8/dense controls match the saved metrics exactly.', '']
    for limited, title in [(0, 'Resident wide words'), (1, 'Supply capped at eight operand pairs per cycle')]:
        lines += ['## '+title, '', 'Command-to-result cycles, shown as **old bank dispatch → merged**.', '',
                  '| Fixture | Scan 8 | Scan 16 | Scan 32 | Scan 64 |', '| --- | ---: | ---: | ---: | ---: |']
        for case in CASES:
            cells = [f'{old[case,m,limited]["command_to_result_cycles"]} → {new[case,m,limited]["command_to_result_cycles"]}' for m in range(4)]
            lines.append('| '+case+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['## Complete-interval utilization at scan 64', '',
              'Useful products / (eight lanes × command-to-result cycles), with resident input.', '',
              '| Fixture | Old utilization | Merged utilization | Cycle speedup |', '| --- | ---: | ---: | ---: |']
    for case in ['half_w8', 'ten_percent_w8', 'point_one_x_99']:
        a, b = old[case,3,0], new[case,3,0]
        ca, cb, useful = int(a['command_to_result_cycles']), int(b['command_to_result_cycles']), int(b['useful_products'])
        lines.append(f'| {case} | {100*useful/(8*ca):.3f}% | {100*useful/(8*cb):.3f}% | {ca/cb:.2f}x |')
    lines += ['', 'Merging removes partial-bank drain cycles. It does not remove original-position scanning or frame-tail overhead. '
              'The extreme sparse fixture is unchanged because it rarely has multiple nonempty banks in a frame. '
              'The resident gains depend on wide local operand availability; the capped experiment still pays acquisition from the first frame. '
              'W4 source-stall fixtures are correctness-only: bubbles are injected per frame and are not comparable across widths.', '',
              '## Isolated synthesis', '',
              '| Width | Design | LUTs | Registers | DSPs | BRAM | Setup WNS ns | Hold WHS ns |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    violating = []
    for width in [8,16,32,64]:
        for folder, name in [('wide_sparse_dot_synth','Bank dispatch'), ('merged_sparse_dot_synth','Merged')]:
            data = synth(ROOT/f'build/{folder}/scan{width}')
            lines.append(f'| {width} | {name} | '+' | '.join(map(str,data))+' |')
            if name == 'Merged' and float(data[4]) < 0:
                violating.append(str(width))
    if violating:
        lines += ['', '**The merged design misses the 10 ns setup target at widths '+', '.join(violating)+'.** '
                  'The combinational prototype demonstrates the cycle benefit but is not ready to claim 100 MHz operation at those widths. '
                  'A registered or staged merger should be evaluated next, including its added latency and backpressure.']
    lines += ['', 'Vivado 2024.2, xc7z020clg400-1, 10 ns clock with 0.300 ns uncertainty before synthesis. '
              'These are unplaced estimates, not measured Fmax or timing closure. No IO delays, physical clock placement, SRAM banking, DMA, DDR loading or board run are included. '
              'A cycle reduction is not yet a hardware-time or energy improvement.', '',
              '[Architecture and reproduction](MERGED_FINDER.md) · [Raw metrics](MERGED_FINDER_PERFORMANCE.csv)', '']
    target = ROOT/'research/MERGED_FINDER_RESULTS.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    shutil.copyfile(raw, ROOT/'research/MERGED_FINDER_PERFORMANCE.csv')
    print(target)


if __name__ == '__main__':
    main()
