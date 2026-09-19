"""Compare the four-stage-DSP 100 MHz baseline to the pipelined 200 MHz engine."""
import csv
from report_tiled_activation import ROOT, physical


def rows(path):
    with path.open(newline='') as f:
        return [{k: v if k == 'fixture' else int(v) for k, v in r.items()
                 if k in ('fixture', 'k', 'rows', 'shift', 'skip', 'reuse', 'activation_bw',
                          'kept', 'occupied_blocks', 'consumer_cycles', 'activation_bytes',
                          'weight_bytes', 'metadata_read_bytes', 'useful_macs')}
                for r in csv.DictReader(f)]


def main():
    base = rows(ROOT / 'research/PIPELINE_200_BASELINE.csv')
    fast = rows(ROOT / 'build/tiled_activation_200/tiled_activation.csv')
    fixed = rows(ROOT / 'build/tiled_activation_200_fixed_bw/tiled_activation.csv')
    assert len(base) == len(fast) == len(fixed) == 76
    for folder in ('tiled_activation_200', 'tiled_activation_200_fixed_bw'):
        log = (ROOT / 'build' / folder / 'xsim.log').read_text()
        assert 'PASS: tiled activation commands=76 tiles=336 dots=2548' in log
        assert 'PASS: producer exhaustive cases=32768' in log
    # Matched fixtures and identical useful work/traffic; only time changes.
    for a, b, c in zip(base, fast, fixed):
        for key in a:
            if key != 'consumer_cycles':
                assert a[key] == b[key] == c[key], (a['fixture'], key)
    consumer = physical('tiled_activation_200_impl', 'pipeline200_impl.log',
                        'PASS: tiled activation registered boundary period=5.000ns',
                        'llm_tiled_activation_timing_boundary', 1072, 8)
    producer = physical('block_map_200', 'pipeline200_producer.log',
                        'PASS: block map registered boundary period=5.000ns',
                        'llm_block_map_timing_boundary', 148, 0)
    audit = (ROOT / 'build/tiled_activation_200_impl/dsp_registers.rpt').read_text()
    for prop in ('AREG=2', 'BREG=2', 'MREG=1', 'PREG=1'):
        assert audit.count(prop) == 8
    text = ['# Pipelined activation engine at 200 MHz', '',
            'The consumer and producer separately meet 5.000 ns with 0.300 ns uncertainty on '
            'xc7z020clg400-1. This qualifies registered block boundaries, not a complete board or DDR system.', '',
            '## Physical results', '',
            '| Unit | LUTs | Fabric FFs | BRAM36 equivalents | DSPs | Setup slack ns | Hold slack ns |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, r in (('Consumer', consumer), ('Producer', producer)):
        text.append(f'| {name} | {r[0]} | {r[1]} | {r[2]:g} | {r[3]} | {r[4]:+.3f} | {r[5]:+.3f} |')
    text += ['', 'The previous 100 MHz consumer used 2497 LUTs / 1542 fabric FFs, eight DSPs '
             'and 2.5 BRAM tiles. The previous producer used 254 LUTs / 143 FFs. '
             'Boundary fixture FFs (1072 / 148) are excluded from these counts. '
             'Both runs have no internal unconstrained timing endpoints, routing errors, or DRC errors; '
             'only the outer fixture ports are excepted.', '',
             '## Matched elapsed-time comparison', '',
             'K=4096, 64 output rows, eight lanes, resident metadata, block skipping and reuse. '
             'Baseline is the immediately preceding four-stage-DSP engine at 100 MHz, not a dense baseline.', '',
             '| Valid density | Old cycles at 100 MHz | New cycles at 200 MHz | Speedup, same bytes/cycle | New cycles, fixed bytes/second | Speedup, fixed bytes/second |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    comparison = []
    for a, b, c in zip(base, fast, fixed):
        if a['fixture'].startswith('density_') and a['rows'] == 64 and a['skip'] and a['reuse']:
            density = int(a['fixture'].split('_')[1])/10
            x, y, z = (r['consumer_cycles'] for r in (a, b, c))
            text.append(f'| {density:g}% | {x} | {y} | {2*x/y:.3f}x | {z} | {2*x/z:.3f}x |')
            comparison.append(dict(density_percent=density, baseline_cycles=x, new_cycles=y,
                                   fixed_bandwidth_cycles=z, baseline_us=x/100,
                                   new_us=y/200, fixed_bandwidth_us=z/200))
    text += ['', 'The same-bytes/cycle model doubles activation and weight bandwidth from '
             '0.8 to 1.6 GB/s each when moving from 100 to 200 MHz. The fixed-bandwidth model '
             'doubles activation assembly/response cycles and spaces weight service by two cycles '
             'with four-cycle minimum response latency, keeping both supplies at 0.8 GB/s. '
             'Both use finite eight-entry response queues. These are simulated local interfaces, not DDR measurements.', '',
             'Only resident-metadata consumer time is compared here. The new full-rate producer needs '
             '515 cycles instead of 513; its source was not bandwidth-throttled in the fixed-bandwidth run. '
             'Cold metadata preparation must therefore be modeled separately before drawing end-to-end conclusions.', '',
             '## What changed', '',
             '- Block request generation separates group selection from eight-bit lane selection and registers the request.',
             '- The finder registers magnitudes, threshold decisions, group occupancy, group payload selection and survivor output. '
             'Local lane traversal sustains one survivor/cycle; group transitions and frame preparation add bubbles.',
             '- The tiled broadcast core uses fixed lane-to-row assignments at elaboration, eliminating the unnecessary general '
             'cross-lane reduction. The generic tagged core retains its original mapping by default.',
             '- Empty-map detection runs after command-map capture in the existing START stage.',
             '- The producer pipelines magnitude, threshold and bitmap updates. The four-stage DSP pipeline remains enabled.', '',
             'No tiles are overlapped: the controller still waits for each result handshake before starting the next tile. '
             'The frequency gain must be weighed against added selection cycles and actual memory bandwidth.', '',
             '## Validation', '',
             'Each memory model passes 76 complete commands, 336 tile results and 2548 dot results, plus '
             '32768 producer threshold cases. Tests include stalls, reset, empty inputs, partial tiles, and full descriptor caches. '
             'The finder additionally passes 4232 frames / 101102 survivors and reset cancellation at seven pipeline depths. '
             'Three shared-activation variants pass 129 vectors / 768 dots. '
             'Physical verification checks DSP registers, setup/hold, exceptions, timing coverage, routing and DRC.', '',
             'Reproduce the simulations and routing using [TILED_ACTIVATION.md](TILED_ACTIVATION.md), then run '
             '`python FPGA/llm/research/report_pipeline_200.py`. Baseline input data is preserved in '
             '[PIPELINE_200_BASELINE.csv](PIPELINE_200_BASELINE.csv).', '',
             '## Device limits', '',
             'AMD DS187 tables 66/67 specify 464.25 MHz for a fully registered DSP48E1 and 464 MHz '
             'for the global clock tree on the -1 speed grade. These component limits are not a promise that '
             'arbitrary logic meets that frequency. Our achieved frequency is determined by routed register-to-register paths. '
             '[AMD DS187](https://docs.amd.com/api/khub/documents/uaBd8Qxmf_K1~~tiVnOxrw/content).', '']
    (ROOT / 'research/PIPELINE_200_RESULTS.md').write_text('\n'.join(text), encoding='utf-8')
    with (ROOT / 'research/PIPELINE_200_COMPARISON.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0]))
        writer.writeheader(); writer.writerows(comparison)
    print('PASS: matched work and traffic across 76 fixtures; both units routed at 200 MHz')


if __name__ == '__main__':
    main()
