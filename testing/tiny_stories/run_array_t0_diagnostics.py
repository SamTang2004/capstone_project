"""Service-based array timing sweep with exact INT32 golden checks.

Clock Hz is a reporting assumption, not hardware frequency discovery.
No bitstream programming, MMIO access, or cache flush is performed.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'FPGA'))
from budgetcuda import Device
from budgetcuda.compiler import GraphBuilder
from budgetcuda.diagnostics import DiagnosticsSettings
from budgetcuda.packing import pack_transformer_activations_int8


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address', default='192.168.2.99')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if args.repeats <= 0:
        parser.error('repeats must be positive')
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  address=args.address, clock_hz_assumed=100000000,
                  clock_verified=False, samples=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = Device(args.address, 9000, 10)
    settings = DiagnosticsSettings()
    rng = np.random.default_rng(17)
    for int4 in (True, False):
        for m in (1, 8, 16):
            for k in (16, 32, 64, 128):
                n = 8
                a = rng.integers(-7, 8, size=(m, k), dtype=np.int32)
                b = rng.integers(-7, 8, size=(k, n), dtype=np.int32)
                golden = a @ b
                builder = GraphBuilder('t0')
                ai = builder.add_transformer_activation('a', m, k)
                bi = builder.add_transformer_weight('b', k, n, b.tolist(), int4=int4)
                ci = builder.add_buffer('c', (m, n), 'int32')
                builder.transformer_gemm(ai, bi, ci, m=m, n=n, k=k)
                model = device.load_model(builder.package(ai, ci))
                failed = False
                try:
                    payload = pack_transformer_activations_int8(a.tolist(), m, k)
                    for iteration in range(args.repeats):
                        print(f"Starting {'W4' if int4 else 'W8'} M={m} N={n} K={k} repeat={iteration}", flush=True)
                        result = model.run(payload, diagnostics=settings)
                        actual = np.asarray(result.output).reshape(m, n)
                        np.testing.assert_array_equal(actual, golden)
                        if len(result.diagnostics.layers) != 1:
                            raise RuntimeError('expected exactly one GEMM diagnostic record')
                        layer = result.diagnostics.layers[0]
                        sample = dict(mode='w4' if int4 else 'w8', m=m, n=n, k=k,
                                      iteration=iteration, golden_pass=True,
                                      diagnostics=asdict(result.diagnostics),
                                      counters=layer.transformer_cycles)
                        report['samples'].append(sample)
                    last = report['samples'][-1]['counters']
                    print(json.dumps(dict(mode=sample['mode'], m=m, k=k, **last)), flush=True)
                    temp = args.output.with_suffix('.tmp')
                    temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
                    temp.replace(args.output)
                except Exception as exc:
                    failed = True
                    report['failure'] = dict(mode='w4' if int4 else 'w8', m=m,
                                             n=n, k=k, error=str(exc),
                                             device_buffers_may_remain_allocated=True)
                    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
                    raise
                finally:
                    if failed:
                        # Do not issue a series of FREE requests against a possibly
                        # wedged device or replace the original exception with cleanup.
                        device.protocol.close()
                    else:
                        model.close()


if __name__ == '__main__':
    main()
