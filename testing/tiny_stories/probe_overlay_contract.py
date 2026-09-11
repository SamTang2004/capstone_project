"""One-operation DeiT-overlay probe; no programming or automatic recovery.

Default is transport readback only. --execute explicitly enables one PL op.
This tests existing integer contracts, not an LLM decoder or QAT equivalence.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from math import isqrt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'FPGA'))
from budgetcuda import Device
from budgetcuda.compiler import GraphBuilder
from budgetcuda.diagnostics import DiagnosticsSettings, parse_diagnostics
from budgetcuda.packing import pack_transformer_activations_int8
from budgetcuda.protocol import ProtocolError


def make_case(case, epsilon=0):
    builder = GraphBuilder('overlay_probe')
    if case == 'layernorm':
        rows = [[-3, -1, 1, 3]] * 3
        a = builder.add_buffer('a', (3, 4), 'int8')
        g = builder.add_buffer('gamma', (4,), 'int16', data=[256] * 4)
        b = builder.add_buffer('beta', (4,), 'int16', data=[0] * 4)
        c = builder.add_transformer_activation('c', 3, 4)
        builder.layer_norm(a, g, b, c, rows=3, features=4, epsilon_term=epsilon)
        golden = []
        for row in rows:
            total = sum(row)
            denominator = max(1, isqrt(4 * sum(x*x for x in row) - total*total + epsilon))
            normalized = []
            for x in row:
                numerator = (4*x-total) << 14
                norm = (abs(numerator) // denominator) * (-1 if numerator < 0 else 1)
                scaled = (norm * 256) >> 14
                normalized.append(max(-128, min(127, (scaled + (128 if scaled >= 0 else -128)) >> 8)))
            golden.append(normalized)
        payload = bytes(x & 255 for row in rows for x in row)
        expected = pack_transformer_activations_int8(golden, 3, 4)
    else:
        rows = [[1, 2], [3, 4], [5, 6]]
        weights = [[7, -3, -2], [4, 5, 3]]
        a = builder.add_transformer_activation('a', 3, 2)
        b = builder.add_transformer_weight('b', 2, 3, weights, int4=case == 'gemm-w4')
        c = builder.add_buffer('c', (3, 3), 'int32')
        builder.transformer_gemm(a, b, c, m=3, n=3, k=2)
        payload = pack_transformer_activations_int8(rows, 3, 2)
        import struct
        expected = struct.pack('<9i', *[sum(row[k]*weights[k][j] for k in range(2)) for row in rows for j in range(3)])
    return builder.package(a, c), payload, expected


def raw(value):
    return value.tobytes() if hasattr(value, 'tobytes') else bytes(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--address', default='192.168.2.99')
    parser.add_argument('--case', choices=('layernorm', 'gemm-w8', 'gemm-w4'), required=True)
    parser.add_argument('--epsilon', type=int, default=0)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package, payload, expected = make_case(args.case, args.epsilon)
    report = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), case=args.case,
                  epsilon=args.epsilon, execute=args.execute, input_hex=payload.hex(),
                  expected_hex=expected.hex(), checks=[])
    device = Device(args.address, 9000, 10)
    model = None
    diagnostics = None
    responsive = True
    success = False
    settings = DiagnosticsSettings(peak_operations_per_cycle=128)
    try:
        model = device.load_model(package)
        device.write(model.input_tensor, payload)
        for index, (buffer, tensor) in enumerate(zip(package.buffers, model.tensors)):
            wanted = payload if tensor is model.input_tensor else buffer.data
            if wanted is None:
                continue
            actual = raw(device.read(tensor))
            passed = actual == bytes(wanted)
            report['checks'].append(dict(buffer=index, passed=passed, actual_hex=actual.hex()))
            if not passed:
                raise RuntimeError(f'Buffer {index} readback mismatch; PL not launched')
        if args.execute:
            initial = settings.initial_buffer(1)
            diagnostics = device.alloc((len(initial),), 'int8', usage='workspace')
            device.write(diagnostics, initial)
            try:
                device.execute_graph(model.descriptor_tensor, model.input_tensor,
                                     model.output_tensor, settings.flags, diagnostics)
            except ProtocolError:
                report['diagnostics'] = asdict(parse_diagnostics(device.read(diagnostics), settings))
                raise
            report['diagnostics'] = asdict(parse_diagnostics(device.read(diagnostics), settings))
            actual = raw(device.read(model.output_tensor))
            report['actual_hex'] = actual.hex()
            report['golden_pass'] = actual == expected
            if actual != expected:
                raise RuntimeError('PL output differs from integer golden')
        success = True
    except Exception as exc:
        report['error'] = str(exc)
        responsive = not isinstance(exc, (TimeoutError, ConnectionError, OSError))
    finally:
        try:
            if responsive:
                if diagnostics is not None:
                    device.free(diagnostics)
                if model is not None:
                    model.close()
        except Exception as exc:
            report['cleanup_error'] = str(exc)
            success = False
        device.protocol.close()
        report['success'] = success
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + '.tmp')
        temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
        temporary.replace(args.output)
    print(json.dumps(report, indent=2))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
