"""Validate and record RTL effects plus routed 100 MHz block timing."""
import re
import shutil
from report_merged_finder import ROOT, CASES, read, rows, key


def implementation(folder, require_timing):
    util = read(folder/'utilization_routed.rpt')
    timing = read(folder/'timing_routed.rpt')
    table = [[c.strip() for c in line.strip().strip('|').split('|')] for line in util.splitlines() if line.strip().startswith('|')]
    header = next(r for r in table if 'Total LUTs' in r and 'FFs' in r)
    def resource_row(instance):
        r = next(r for r in table if len(r) == len(header) and r[0] == instance)
        d = dict(zip(header, r))
        dsp_key = next(k for k in header if 'DSP' in k)
        bram = sum(float(d[k]) for k in header if 'RAMB' in k)
        return [int(d['Total LUTs']), int(d['FFs']), int(d[dsp_key]), int(bram)]
    resource = resource_row('dut')
    total = resource_row('llm_sparse_dot_timing_boundary')
    dut_row = next(r for r in table if len(r) == len(header) and r[0] == 'dut')
    lutram = int(dut_row[header.index('LUTRAMs')])
    slacks = [float(re.search(pattern, timing)[1]) for pattern in [
        r'Slack \([^)]+\)\s*:\s*([-\d.]+)ns\s*\(required time - arrival time\)',
        r'Slack \([^)]+\)\s*:\s*([-\d.]+)ns\s*\(arrival time - required time\)']]
    assert resource[2] == 8
    if require_timing:
        assert min(slacks) >= 0, ('Routed timing failed', folder, slacks)
    for check in ['no_clock', 'unconstrained_internal_endpoints', 'loops', 'latch_loops']:
        assert re.search(r'checking '+check+r' \(0\)', timing), (folder, check)
    route = read(folder/'route_status.rpt')
    assert re.search(r'# of nets with routing errors\.+\s*:\s*0\s*:', route)
    assert total[1]-resource[1] == 1938, ('Boundary register count changed', total, resource)
    exceptions = [line for line in read(folder/'exceptions.rpt').splitlines() if re.match(r'^\s*\d+\s+', line)]
    assert len(exceptions) == 2
    assert any('[get_ports fixture_inputs*]' in line for line in exceptions)
    assert any('[get_ports fixture_outputs*]' in line for line in exceptions)
    assert not re.search(r'\|\s*(Error|Critical Warning)\s*\|', read(folder/'drc.rpt'))
    return resource, slacks, total, lutram


def main():
    unit = re.search(r'PASS: wide finder frames=(\d+) chunks=(\d+) cycles=(\d+) stalls=(\d+) replacements=(\d+)',
                     read(ROOT/'build/pipelined_finder/xsim.log'))
    end = re.search(r'PASS: wide sparse dot8 real DSP48E1 checked=(\d+) cycles=(\d+)',
                    read(ROOT/'build/pipelined_sparse_dot/xsim.log'))
    assert unit and end
    raw = ROOT/'build/pipelined_sparse_dot/wide_performance.csv'
    records = rows(raw)
    pipe = {key(r): r for r in records}
    merged = {key(r): r for r in rows(ROOT/'research/MERGED_FINDER_PERFORMANCE.csv')}
    banked = {key(r): r for r in rows(ROOT/'research/WIDE_FINDER_PERFORMANCE.csv')}
    assert pipe.keys() == merged.keys() == banked.keys()
    for k, r in pipe.items():
        for field in ['source_slots', 'useful_products', 'declared_jobs', 'source_packets', 'supply_wait_cycles']:
            assert r[field] == merged[k][field] == banked[k][field], (k, field)
        if k[1] == 4:
            assert r == merged[k] == banked[k], ('Dense control changed', k)
    for case in CASES:
        for m in range(4):
            assert int(pipe[case,m,0]['command_to_result_cycles']) - int(merged[case,m,0]['command_to_result_cycles']) == m+2
    physical = {name: implementation(ROOT/f'build/{name}_registered_impl/scan64', name == 'pipelined') for name in ['banked','pipelined']}
    lines = ['# Pipelined merger: measured effects', '',
             '**The 64-position block meets the 100 MHz routed timing target between explicit shared-clock launch/capture registers.** '
             'The computation still uses eight DSP48E1 multipliers. This is static timing evidence for the routed out-of-context block, not a board measurement.', '',
             f'RTL correctness: **{end[1]} result vectors / {sum(int(r["declared_jobs"]) for r in records)} dot results** pass in {int(end[2]):,} scoreboard cycles. '
             f'The standalone finder passes **{int(unit[1]):,} frames** / {int(unit[3]):,} cycles, including {int(unit[4]):,} output-stall cycles. '
             'Directed checks fill the pipeline, drain it, reset multiple in-flight frames, reset mid-frame, and sustain 100 consecutive sparse frames without admission bubbles. '
             'Payload, source index, metadata, empty-frame handling and full nonfinal chunks are checked.', '',
             '## Routed timing and area at scan 64', '',
             '| Design | LUTs | Registers | DSPs | BRAM | LUTRAMs included in LUTs | Setup WNS ns | Hold WHS ns |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, (resources, slacks, total, lutram) in physical.items():
        lines.append('| '+name+' | '+' | '.join(map(str, resources))+f' | {lutram} | {slacks[0]:.3f} | {slacks[1]:.3f} |')
    if min(physical['banked'][1]) < 0:
        lines += ['', 'The original banked control fails this physical timing contract; its cycle counts below remain RTL comparisons and do not establish its operation at 100 MHz.']
    lines += ['', 'The table reports DUT resources. The timing harness adds 1,938 preserved launch/capture registers around each design (1,382 input and 556 output bits). '
              'These fixture registers are excluded from DUT cycle and resource counts but included during physical implementation.', '',
              'Vivado 2024.2, xc7z020clg400-1; 10.000 ns clock, 0.300 ns uncertainty, clock origin BUFGCTRL_X0Y0. '
              'All DUT inputs and outputs connect to explicit registers on the same clock. Two false-path constraints exclude only the outer fixture-port wiring, '
              'before launch registers and after capture registers. No DUT or register-to-register path is excepted and no multicycle exception is used. '
              'Timing checks report no missing clocks, unconstrained internal endpoints or loops; all routable nets are fully routed. '
              'The 1,382 missing input-delay and 556 missing output-delay warnings concern only the outer fixture ports. '
              'There are no DRC errors; DSP pipelining advisories and the standalone Zynq PS7 warning remain. '
              'Only width 64 is physically qualified here; the other widths are RTL-tested.', '',
              'The previous combinational width-64 merger had 13,313 LUTs, 2,703 registers and −6.724 ns setup slack in synthesis with no IO budgets. '
              'That older estimate is not a routed comparison. The banked and pipelined rows above use the same physical flow and constraints.', '',
              'Before the two-frame input FIFO, the pipelined design failed a different external-port timing experiment: setup −1.002 ns and hold −1.220 ns '
              'with 2 ns maximum / 0.5 ns minimum IO delays. The FIFO breaks the buffer-to-input-ready path. Explicit shared-clock boundary flops replace '
              'the ideal external-clock IO model in the final experiment. **The earlier external-port contract is not claimed to pass.** '
              'The final result qualifies the registered on-chip block environment; it does not qualify the timing harness fixture ports or board IO.', '']
    for limited, title in [(0, 'Resident wide words'), (1, 'Supply capped at eight operand pairs per cycle')]:
        lines += ['## '+title, '', 'Complete command-to-result cycles at scan 64.', '',
                  '| Fixture | Banked | Combinational merge | Pipelined merge | Pipelined time at 100 MHz | Useful utilization |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for case in CASES:
            r = pipe[case,3,limited]
            c, useful = int(r['command_to_result_cycles']), int(r['useful_products'])
            lines.append(f'| {case} | {banked[case,3,limited]["command_to_result_cycles"]} | '
                         f'{merged[case,3,limited]["command_to_result_cycles"]} | {c} | {c/100:.2f} µs | {100*useful/(8*c):.3f}% |')
        lines.append('')
    area_ratio = physical['pipelined'][0][0]/physical['banked'][0][0]
    lines += ['## Cost versus benefit', '',
              f'The pipelined DUT uses **{area_ratio:.2f}x the LUTs** of the banked control. '
              'The following ratios use the complete resident-input command interval at the same 100 MHz clock; they are not memory-system or energy measurements.', '',
              '| Density fixture | Speedup versus banked | Throughput per LUT versus banked |', '| --- | ---: | ---: |']
    for case in ['half_w8', 'ten_percent_w8', 'point_one_x_99']:
        speed = int(banked[case,3,0]['command_to_result_cycles'])/int(pipe[case,3,0]['command_to_result_cycles'])
        lines.append(f'| {case} | {speed:.2f}x | {speed/area_ratio:.2f}x |')
    lines += ['', 'LUT-normalized throughput omits register, DSP, memory-bandwidth and energy costs. '
              'It is a limited screening metric, not a complete hardware-efficiency claim.', '', '## Pipeline effects', '',
              '- At scan 64, the input FIFO, three registered merge levels and registered serializer add **five cycles / 50 ns** to the unstalled fixtures relative to combinational merging. '
              'Scan 8/16/32 add two/three/four cycles. This is a per-wave fill/drain increase in these fixtures, not a five-cycle penalty on every frame.',
              '- Frames overlap across elastic stages. A frame with eight survivors can be accepted every cycle; a dense 64-entry frame still needs eight serializer cycles. '
              'Backpressure fills the finite pipeline and then stalls input without losing ownership or data.',
              '- More registers store frames in flight. The staged merge tree also replaces the large global rank-selection network, so the area change reflects both added pipeline storage and a different compaction topology.',
              '- Operand frames use distributed RAM in the input queue. Metadata is kept in flip-flops to avoid a short LUTRAM-output hold path. This storage choice does not change cycle latency.',
              '- The input supply ceiling remains. The resident-mode gains require wide local operand availability; the eight-pair cap removes most gains. '
              'The extreme sparse fixture remains below 1% useful utilization.',
              '- No power or energy result is claimed. Additional registers and clock load must be measured against saved arithmetic activity. '
              'No SRAM banking, DMA/DDR path or board validation is included.', '',
              'W4 source-stall cases are correctness-only across widths because source bubbles are per frame. '
              'The report checks equal work, source packet sizes and acquisition charges, and an unchanged dense control.', '',
              '[Architecture and reproduction](PIPELINED_FINDER.md) · [Raw metrics](PIPELINED_FINDER_PERFORMANCE.csv)', '']
    target = ROOT/'research/PIPELINED_FINDER_RESULTS.md'
    target.write_text('\n'.join(lines), encoding='utf-8')
    shutil.copyfile(raw, ROOT/'research/PIPELINED_FINDER_PERFORMANCE.csv')
    print(target)


if __name__ == '__main__':
    main()
