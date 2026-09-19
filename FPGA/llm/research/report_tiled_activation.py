"""Validate RTL traffic/results and physical artifacts for the tiled experiment."""
import csv
import re
from report_activation_sharing_rtl import ROOT, read, slack


def physical(folder_name, log_name, marker, top_name, fixture_ffs, expected_dsps):
    folder = ROOT / 'build' / folder_name
    text = read(folder / 'timing_routed.rpt')
    setup, hold = slack(text)
    assert min(setup, hold) >= 0
    assert 'period 5.000' in read(folder / 'clock.xdc')
    assert marker in read(ROOT / 'build' / log_name)
    for check in ('no_clock', 'unconstrained_internal_endpoints', 'loops', 'latch_loops'):
        assert re.search(r'checking '+check+r' \(0\)', text)
    route = read(folder / 'route_status.rpt')
    assert re.search(r'# of nets with routing errors\.+\s*:\s*0\s*:', route)
    assert not re.search(r'\|\s*(Error|Critical Warning)\s*\|', read(folder / 'drc.rpt'))
    exceptions = [l for l in read(folder / 'exceptions.rpt').splitlines() if re.match(r'^\s*\d+\s+', l)]
    assert len(exceptions) == 2
    assert any('[get_ports fixture_inputs*]' in l for l in exceptions)
    assert any('[get_ports fixture_outputs*]' in l for l in exceptions)
    table = [[v.strip() for v in l.strip().strip('|').split('|')]
             for l in read(folder / 'utilization_routed.rpt').splitlines() if l.strip().startswith('|')]
    dut = next(r for r in table if r[0] == 'dut')
    top = next(r for r in table if r[0] == top_name)
    assert int(top[6]) - int(dut[6]) == fixture_ffs
    assert int(dut[9]) == expected_dsps
    return [int(dut[2]), int(dut[6]), int(dut[7]) + int(dut[8])/2, int(dut[9]), setup, hold]


def main():
    log = read(ROOT / 'build/tiled_activation_200/xsim.log')
    assert 'PASS: producer exhaustive cases=32768' in log
    match = re.search(r'PASS: tiled activation commands=(\d+) tiles=(\d+) dots=(\d+)', log)
    assert match
    with (ROOT / 'build/tiled_activation_200/tiled_activation.csv').open(newline='') as f:
        rows = [{k: v if k == 'fixture' else int(v) for k, v in r.items()} for r in csv.DictReader(f)]
    assert len(rows) == int(match[1])
    assert sum((r['rows']+7)//8 for r in rows) + 1 == int(match[2])
    assert sum(r['rows'] for r in rows) + 8 == int(match[3])
    work = {}
    for r in rows:
        tiles = (r['rows']+7)//8
        passes = 1 if r['reuse'] else tiles
        assert r['activation_requests'] == passes * (r['occupied_blocks'] if r['skip'] else (r['k']+63)//64)
        assert r['selected'] == passes * r['kept']
        assert r['cache_writes'] == (r['kept'] if r['reuse'] else 0)
        assert r['cache_reads'] == ((tiles-1)*r['kept'] if r['reuse'] else 0)
        assert r['weight_bytes'] == r['rows'] * r['kept']
        assert r['producer_cycles'] == 3 + max(1, (r['k']+7)//8)
        assert r['metadata_read_bytes'] == (8 if r['skip'] else 0)
        key = (r['fixture'], r['k'], r['rows'], r['shift'], r['activation_bw'])
        signature = (r['kept'], r['occupied_blocks'], r['weight_bytes'], r['useful_macs'])
        assert work.setdefault(key, signature) == signature
        r['separate_pass_cycles'] = r['consumer_cycles'] + (r['producer_cycles'] if r['skip'] else 0)
        r['separate_pass_activation_bytes'] = r['activation_bytes'] + (r['k'] if r['skip'] else 0)
        r['producer_metadata_write_bytes'] = 8 if r['skip'] else 0
        r['cache_write_bits'] = 20*r['cache_writes']
        r['cache_read_bits'] = 20*r['cache_reads']
        r['useful_utilization'] = r['useful_macs'] / (8*r['consumer_cycles'])
    def select(density, skip=0, reuse=0, nrows=64, fixture=None):
        return next(r for r in rows if r['fixture'] == (fixture or f'density_{density}')
                    and r['rows'] == nrows and r['skip'] == skip and r['reuse'] == reuse)
    consumer = physical('tiled_activation_200_impl','pipeline200_impl.log',
                        'PASS: tiled activation registered boundary period=5.000ns',
                        'llm_tiled_activation_timing_boundary',1072,8)
    producer = physical('block_map_200','pipeline200_producer.log',
                        'PASS: block map registered boundary period=5.000ns','llm_block_map_timing_boundary',148,0)
    assert consumer[2] == 2.5 and producer[2] == 0
    out = ROOT / 'research'
    with (out / 'TILED_ACTIVATION_PERFORMANCE.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines = ['# Block skipping and multi-tile reuse: Vivado results', '',
             f'**Both optimizations are implemented and verified.** {match[1]} complete matrix-vector commands, '
             f'{match[2]} tile results and {match[3]} dot results pass DSP48E1 simulation. The totals include one '
             'correct tile before an intentional replay reset. The producer also passes 32,768 exhaustive '
             'byte/lane/threshold cases. [Architecture and contracts](TILED_ACTIVATION.md).', '',
             '## Latency for 64 output rows', '',
             'At 200 MHz: K=4096, eight compute lanes, eight-byte/cycle activation supply, nonzero signed W8 weights and '
             'deterministic random fixed-count activation masks. Every mode computes all 64 rows. Resident-map '
             'latencies include a one-cycle/eight-byte metadata read. Cold-map latencies also charge the full '
             '515-cycle metadata-building pass. These interfaces scale bytes/second with clock; '
             'see [fixed-bandwidth comparison](PIPELINE_200_RESULTS.md) for the separate constant-bandwidth experiment.', '',
             '| Nominal density | Raw rescan per tile | Reuse only | Skip + reuse, resident map | Resident speedup | Skip + reuse, separate metadata pass |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for d in (1000,500,100,10,1,0):
        a,b,c=select(d),select(d,0,1),select(d,1,1)
        lines.append(f'| {d/10:g}% | {a["consumer_cycles"]} | {b["consumer_cycles"]} | {c["consumer_cycles"]} | '
                     f'{a["consumer_cycles"]/c["consumer_cycles"]:.2f}x | {c["separate_pass_cycles"]} |')
    sparse = select(1,1,1)
    lines += ['', '**Metadata preparation is charged separately.** '
              f'For 0.1% density, resident-map skipping plus reuse takes {sparse["consumer_cycles"]} cycles, but building the map first '
              f'takes {sparse["separate_pass_cycles"]} total cycles. Reuse without a map takes {select(1,0,1)["consumer_cycles"]}. Generate summaries alongside existing '
              'activation production, or amortize their construction over further consumers.', '',
              '## Four-way ablation at 0.1% / 64 rows', '',
              '| Skip | Reuse | Cycles, resident map | Activation bytes | Metadata read bytes | Weight bytes |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    for skip,reuse in ((0,0),(1,0),(0,1),(1,1)):
        r=select(1,skip,reuse)
        lines.append(f'| {bool(skip)} | {bool(reuse)} | {r["consumer_cycles"]} | {r["activation_bytes"]} | '
                     f'{r["metadata_read_bytes"]} | {r["weight_bytes"]} |')
    lines += ['', 'These are activation-source reads, not all memory traffic. The combined mode also writes '
              'four 20-bit descriptors and reads them 28 times for later tiles (80 write bits / 560 read bits). '
              'A cold metadata pass adds 4096 activation read bytes and an eight-byte metadata write. '
              'No energy or DDR-transaction reduction is inferred from these counts.', '',
              '## Single-tile and distribution limits', '',
              '| Case | Raw baseline cycles | Combined resident cycles | Combined cold-map cycles |',
              '| --- | ---: | ---: | ---: |']
    for d in (100,10,1):
        a,c=select(d,nrows=8),select(d,1,1,8)
        lines.append(f'| Eight rows, {d/10:g}% density | {a["consumer_cycles"]} | {c["consumer_cycles"]} | {c["separate_pass_cycles"]} |')
    a=select(10,fixture='clustered_one_percent');c=select(10,1,1,fixture='clustered_one_percent')
    lines.append(f'| 64 rows, clustered 1% | {a["consumer_cycles"]} | {c["consumer_cycles"]} | {c["separate_pass_cycles"]} |')
    lines += ['', 'The random 1% fixture occupies 29 blocks; its clustered counterpart occupies one. '
              'The same survivor count therefore gives different skip benefits. Dense/high-density inputs '
              'occupy every block, so summaries save no reads and incur their metadata access.', '',
              '## Physical implementation', '',
              'Both units separately pass **200 MHz** on xc7z020clg400-1 with 0.300 ns clock uncertainty '
              'between explicit shared-clock boundary registers.', '',
              '| Unit | Routed LUTs | FFs | BRAM36-equivalent tiles | DSPs | Setup slack ns | Hold slack ns |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,a in (('Multi-tile consumer',consumer),('Metadata producer',producer)):
        lines.append(f'| {name} | {a[0]} | {a[1]} | {a[2]:g} | {a[3]} | {a[4]:+.3f} | {a[5]:+.3f} |')
    lines += ['', f'The consumer grows from the previous one-tile streaming block\'s 1,735 LUTs / 1,325 FFs / '
              f'zero BRAM to {consumer[0]} LUTs / {consumer[1]} FFs / 2.5 BRAM tiles. The producer costs '
              f'an additional {producer[0]} LUTs / {producer[1]} FFs. These are separately placed blocks, '
              'not an integrated system resource/timing result. Activation/weight memories, their queues, '
              'metadata storage between units, DMA and DDR remain outside these counts.', '',
              'The consumer and producer timing fixtures add 1072 and 148 preserved boundary FFs, respectively, '
              'excluded from DUT area and functional cycle counts. Only outer fixture-port paths are excepted. '
              'Both have no internal missing clocks, unconstrained endpoints, loops, routing errors or DRC errors. '
              'DSP pipelining/standalone PS7 warnings and expected outer-port delay warnings are not board IO qualification.', '',
              '## Interpretation', '',
              '- These results use the four-stage DSP product pipeline, pipelined bitmap selection and producer, '
              'and fixed lane-to-row arithmetic. See [200 MHz comparison](PIPELINE_200_RESULTS.md) for the 100 MHz baseline.',
              '- Reuse removes repeated scans after tile 0. Block summaries remove empty-block reads from that '
              'first scan when metadata is already available. They address different costs.',
              '- At high density, scanning overlaps useful computation; caching saves activation-source reads '
              'but gives little latency gain. It still consumes BRAM and descriptor traffic.',
              '- At extreme sparsity, startup/drain for each eight-row tile becomes a substantial residual cost. '
              'The prototype deliberately waits for each result handshake before starting the next tile.',
              '- No quality claim for positive pruning thresholds, no power measurement, no board execution, '
              'and no full LLM inference or end-to-end memory-system speedup claim.', '']
    (out / 'TILED_ACTIVATION_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'PASS: {len(rows)} commands, {match[2]} tiles, {match[3]} dots, 32768 producer cases; both routed at 200MHz')


if __name__=='__main__':
    main()
