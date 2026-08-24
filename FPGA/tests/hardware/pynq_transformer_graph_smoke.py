"""Board-level Transformer graph scheduler smoke test."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda import Device, build_deit_attention_block_package, build_deit_mlp_block_package
from budgetcuda.diagnostics import DiagnosticsSettings, parse_diagnostics, summarize_cycle_report


def _values(result) -> list[int]:
    return result.reshape(-1).tolist() if hasattr(result, "reshape") else list(result)


def _run_with_diagnostics(model, input_values):
    settings = DiagnosticsSettings()
    initial = settings.initial_buffer(len(model.package.layers))
    diagnostics = model.device.alloc((len(initial),), "int8", usage="workspace")
    try:
        model.device.write(model.input_tensor, input_values)
        model.device.write(diagnostics, initial)
        try:
            model.device.execute_graph(
                model.descriptor_tensor, model.input_tensor, model.output_tensor,
                settings.flags, diagnostics,
            )
        except Exception:
            report = parse_diagnostics(model.device.read(diagnostics), settings)
            summary = ", ".join(f"{layer.layer_index}:{layer.op_name}={layer.status}" for layer in report.layers)
            print(f"failed graph layers: {summary or 'none entered'}")
            expected = model.package.serialize_descriptor([tensor.buffer_id for tensor in model.tensors])
            actual = bytes(model.device.read(model.descriptor_tensor))
            differences = [index for index, (lhs, rhs) in enumerate(zip(expected, actual)) if lhs != rhs]
            if differences:
                first, last = differences[0], differences[-1]
                print(f"descriptor corruption bytes={len(differences)} range={first}..{last}")
                print(f"expected[{first}:{min(first + 32, len(expected))}]={expected[first:first + 32].hex()}")
                print(f"actual[{first}:{min(first + 32, len(actual))}]={actual[first:first + 32].hex()}")
            raise
        report = parse_diagnostics(model.device.read(diagnostics), settings)
        breakdown = summarize_cycle_report(report)["module_cycle_breakdown"]
        print("module cycles: " + ", ".join(
            f"{item['module']}={item['window_cycles']} "
            f"({item['total_window_percent']:.2f}%)"
            for item in breakdown
        ))
        return model.device.read(model.output_tensor)
    finally:
        model.device.free(diagnostics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--mode", choices=("w8", "w4", "both"), default="both")
    args = parser.parse_args()

    device = Device(args.address, args.port, args.timeout)
    modes = (False, True) if args.mode == "both" else (args.mode == "w4",)
    for int4 in modes:
        label = "W4A8" if int4 else "W8A8"

        mlp_input = [-3, -1, 1, 3] * 3
        mlp = device.load_model(build_deit_mlp_block_package(
            3, 4, 8, [0] * 32, [0] * 32,
            accelerator_layout=True, weight_int4=int4,
        ))
        try:
            print(f"{label} MLP buffers=" + ",".join(str(t.buffer_id) for t in mlp.tensors) +
                  f" descriptor={mlp.descriptor_tensor.buffer_id} input={mlp.input_tensor.buffer_id} output={mlp.output_tensor.buffer_id}")
            actual = _values(_run_with_diagnostics(mlp, mlp_input))
            if actual != mlp_input:
                raise RuntimeError(f"{label} MLP residual mismatch: {actual}")
        finally:
            mlp.close()

        attention_input = [-3, -2, -1, 1, 2, 3] * 2
        zero_weights = [0] * 36
        attention = device.load_model(build_deit_attention_block_package(
            2, 6, 3, zero_weights, zero_weights, zero_weights, zero_weights,
            weight_int4=int4,
        ))
        try:
            print(f"{label} attention buffers=" + ",".join(str(t.buffer_id) for t in attention.tensors) +
                  f" descriptor={attention.descriptor_tensor.buffer_id} input={attention.input_tensor.buffer_id} output={attention.output_tensor.buffer_id}")
            actual = _values(_run_with_diagnostics(attention, attention_input))
            if actual != attention_input:
                raise RuntimeError(f"{label} attention residual mismatch: {actual}")
        finally:
            attention.close()

        print(f"PASS: {label} MLP and complete three-head attention graphs")


if __name__ == "__main__":
    main()
