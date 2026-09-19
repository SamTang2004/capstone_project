"""Check Vivado artifacts and report actual activation-sharing RTL measurements."""
from pathlib import Path
import csv
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research'
NAMES = ['Tree cached', 'Bitmap cached', 'Bitmap streaming']
BITS = [(0, 0), (1, 0), (1, 1)]


def read(path):
    return path.read_text(encoding='utf-8', errors='replace')


def slack(text):
    return [float(re.search(pattern, text)[1]) for pattern in (
        r'Slack \([^)]+\)\s*:\s*([-\d.]+)ns\s*\(required time - arrival time\)',
        r'Slack \([^)]+\)\s*:\s*([-\d.]+)ns\s*\(arrival time - required time\)')]


def resources(text):
    def value(label):
        return float(re.search(r'^\|\s*' + re.escape(label) + r'\s*\|\s*([\d.]+)', text, re.M)[1])
    return [int(value('Slice LUTs*')), int(value('Slice Registers')),
            value('Block RAM Tile'), int(value('DSPs'))]


def main():
    rows = []
    checked = dots = 0
    for mode, (bitmap, streaming) in enumerate(BITS):
        log = read(ROOT / f'build/activation_sharing/simulation_{mode}.log')
        match = re.search(rf'PASS: activation sharing BITMAP={bitmap} STREAMING={streaming} vectors=(\d+) dots=(\d+)', log)
        assert match, f'Missing integration PASS mode {mode}'
        checked += int(match[1]); dots += int(match[2])
        with (ROOT / f'build/activation_sharing/activation_sharing_{bitmap}_{streaming}.csv').open(newline='') as f:
            cases = [{k: v if k == 'fixture' else int(v) for k, v in row.items()} for row in csv.DictReader(f)]
        assert len(cases) == int(match[1])
        for r in cases:
            assert r['bitmap'] == bitmap and r['streaming'] == streaming
            assert r['descriptor_writes'] == r['passes'] * r['kept']
            assert r['weight_bytes'] == r['rows'] * r['kept']
            assert r['scan_frames'] == r['passes'] * max(1, (r['k'] + 63) // 64)
            assert 0 <= r['useful_macs'] <= r['weight_bytes']
            assert r['cycles'] >= r['scan_cycles']
            if r['fixture'] == 'signed_threshold_stalls':
                assert r['result_stalls'] == 7
            r['variant'] = NAMES[mode]
            r['useful_utilization'] = r['useful_macs'] / (8 * r['cycles'])
            rows.append(r)
    unit = re.search(r'PASS: bitmap finder frames=(\d+) survivors=(\d+) stalls=(\d+)', read(ROOT/'build/bitmap_finder/xsim.log'))
    assert unit
    # Identical fixture work across architectures and repeated/shared execution.
    groups = {}
    for r in rows:
        key = (r['fixture'], r['k'], r['rows'], r['shift'], r['activation_bw'])
        work = (r['kept'], r['weight_bytes'], r['useful_macs'])
        assert groups.setdefault(key, work) == work
    synth_log = read(ROOT/'build/activation_sharing_synth.log')
    area = []
    for mode in range(3):
        assert f'PASS: activation sharing synthesis MODE={mode}' in synth_log
        folder = ROOT / f'build/activation_sharing_synth/mode{mode}'
        a = resources(read(folder/'utilization.rpt'))
        assert a[3] == 8
        assert a[2] == (0 if mode == 2 else 2.5)
        area.append(a + slack(read(folder/'timing_synth.rpt')))
    with (OUT/'ACTIVATION_SHARING_RTL_PERFORMANCE.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    def select(mode, density, bw=8, passes=1):
        return next(r for r in rows if r['variant'] == NAMES[mode] and r['fixture'] == f'density_{density}'
                    and r['activation_bw'] == bw and r['passes'] == passes)
    lines = ['# Activation sharing: Vivado results', '',
             f'RTL simulation passes **{checked} result vectors / {dots} dot products** across three architectures, '
             f'using DSP48E1 models. The standalone bitmap finder passes **{unit[1]} frames / {unit[2]} survivors**, '
             f'including {unit[3]} stalled-output cycles. [Architecture and scope](ACTIVATION_SHARING_RTL.md).', '',
             '## Shared selection: command-to-result cycles', '',
             'K=4096, eight output rows, signed W8 nonzero weights, deterministic random fixed-count activation '
             'masks. Eight activation bytes/cycle; eight output-interleaved weight banks, two-cycle responses. '
             'Acquisition, selection and arithmetic drain are included. No source/result stalls in this table.', '',
             '| Nominal activation density | Survivors | Tree cached | Bitmap cached | Bitmap streaming | Streaming useful utilization |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for density in (1000, 500, 100, 10, 1, 0):
        a, b, c = [select(m, density) for m in range(3)]
        lines.append(f'| {density/10:g}% | {c["kept"]} | {a["cycles"]} | {b["cycles"]} | {c["cycles"]} | {c["useful_utilization"]:.2%} |')
    lines += ['', '## Scanning efficiency of bitmap streaming', '',
              'The bound is max(ceil(K/activation_bandwidth), survivor_count). One selected activation per cycle '
              'can occupy all eight compute lanes. Scan cycles include input acquisition, finite-buffer backpressure '
              'and frame handling. Efficiency here is relative to that service bound, not the fraction of inputs retained.', '',
              '| Nominal density | Scan cycles | Optimistic bound | Bound / scan cycles | Input positions scanned/cycle |',
              '| --- | ---: | ---: | ---: | ---: |']
    for density in (1000, 500, 100, 10, 1):
        c = select(2, density)
        bound = max(512, c['kept'])
        lines.append(f'| {density/10:g}% | {c["scan_cycles"]} | {bound} | {bound/c["scan_cycles"]:.2%} | {4096/c["scan_cycles"]:.3f} |')
    lines += ['', '## Sharing ablation at 10% nominal density', '',
              'Eight passes repeat classification but compute the output tile only once. The same weight requests '
              'and arithmetic are used in both columns. This is a repeated-scanning ablation, not a dense baseline.', '',
              '| Variant | Eight scans | One shared scan | Speedup |', '| --- | ---: | ---: | ---: |']
    for mode in range(3):
        a, b = select(mode, 100, passes=8), select(mode, 100)
        lines.append(f'| {NAMES[mode]} | {a["cycles"]} | {b["cycles"]} | {a["cycles"]/b["cycles"]:.2f}x |')
    lines += ['', '## Synthesis comparison', '',
              'Same part, 10 ns clock and 0.300 ns uncertainty. These are **unplaced** LUT/register counts and '
              'timing estimates; external IO paths lack delay budgets. Only the separately routed streaming '
              'qualification below can establish the stated clock target.', '',
              '| Variant | LUTs | FFs | BRAM36-equivalent tiles | DSPs | Setup estimate ns | Hold estimate ns |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, a in zip(NAMES, area):
        lines.append(f'| {name} | {a[0]} | {a[1]} | {a[2]:g} | {a[3]} | {a[4]:+.3f} | {a[5]:+.3f} |')
    folder = ROOT/'build/activation_sharing_registered_impl'
    physical = read(folder/'timing_routed.rpt')
    setup, hold = slack(physical)
    assert min(setup, hold) >= 0, (setup, hold)
    assert 'PASS: shared activation streaming registered boundary 100MHz' in read(ROOT/'build/activation_sharing_impl.log')
    for check in ('no_clock', 'unconstrained_internal_endpoints', 'loops', 'latch_loops'):
        assert re.search(r'checking '+check+r' \(0\)', physical)
    assert re.search(r'# of nets with routing errors\.+\s*:\s*0\s*:', read(folder/'route_status.rpt'))
    assert not re.search(r'\|\s*(Error|Critical Warning)\s*\|', read(folder/'drc.rpt'))
    exceptions = [line for line in read(folder/'exceptions.rpt').splitlines() if re.match(r'^\s*\d+\s+', line)]
    assert len(exceptions) == 2 and all('fixture_' in line for line in exceptions)
    hierarchy = [[v.strip() for v in line.strip().strip('|').split('|')]
                 for line in read(folder/'utilization_routed.rpt').splitlines() if line.strip().startswith('|')]
    dut = next(r for r in hierarchy if r[0] == 'dut')
    top = next(r for r in hierarchy if r[0] == 'llm_shared_activation_timing_boundary')
    assert int(top[6]) - int(dut[6]) == 983
    assert int(dut[9]) == 8
    lines += ['', '## Routed 100 MHz qualification', '',
              f'**Bitmap streaming passes** at 10.000 ns with 0.300 ns clock uncertainty: '
              f'setup slack **{setup:+.3f} ns**, hold slack **{hold:+.3f} ns**. '
              f'Routed DUT: **{dut[2]} LUTs / {dut[6]} FFs / {dut[9]} DSPs**, '
              f'{dut[7]} RAMB36 and {dut[8]} RAMB18.', '',
              'All DUT ports connect to same-clock launch/capture registers. The 983 preserved fixture registers '
              'are excluded from DUT area and functional cycle counts, but included in placement. Only outer '
              'fixture-port paths are excepted. No DUT paths or register-to-register paths are false or multicycle. '
              'The expected unconstrained outer-port delay warnings do not qualify board IO. No internal missing '
              'clocks/unconstrained endpoints/loops, routing errors or DRC errors are present.', '',
              'This qualifies the streaming on-chip block boundary. It does not include physical weight RAM, '
              'its response queue, DMA/DDR, production placement or board validation. The two cached variants '
              'have synthesis estimates only.', '', '## Implications and limits', '',
              f'- Bitmap cached uses {1-area[1][0]/area[0][0]:.1%} fewer synthesized LUTs than tree cached '
              'for a similar cycle count. Both include the same descriptor capacity and eight DSPs.',
              '- Streaming overlaps scanning and computation; it removes the descriptor cache for one eight-row tile. '
              'Retaining/replaying selections for later output tiles is future work.',
              f'- At 64 activation bytes/cycle, bitmap streaming takes {select(2,100,64)["cycles"]} cycles '
              f'at 10% density ({select(2,100,64)["useful_utilization"]:.2%} useful utilization). '
              'This gain requires that wide input supply.',
              '- Near 0.1% density the eight-byte source supplies only about 0.008 useful activations/cycle. '
              'The low MAC utilization reflects input sparsity and bandwidth even when scanning is nearly optimal.',
              '- The Python model used different buffering and a different preparation throughput. These RTL '
              'numbers supersede its estimates only for this concrete eight-row, output-interleaved configuration.',
              '- No power measurement or energy claim. No general sparse-weight compaction, multi-tile RTL '
              'reuse, real LLM quality evaluation or external-memory performance claim.', '']
    (OUT/'ACTIVATION_SHARING_RTL_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'PASS: report checked {checked} vectors, {dots} dots, {unit[1]} bitmap frames, synthesis and routed timing')


if __name__ == '__main__':
    main()
