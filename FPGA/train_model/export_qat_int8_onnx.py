"""Export the trained apple/banana/orange QAT checkpoint as Q/DQ INT8 ONNX."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import onnx
import numpy as np
import torch

from export_qat_quant_params import load_converted_model_from_qat
from paths import QAT_CHECKPOINT, QAT_MODEL_DIR


DEFAULT_OUTPUT = QAT_MODEL_DIR / "fruit3_qat_int8.onnx"


def export_qat_int8_onnx(checkpoint: Path, output: Path) -> dict[str, object]:
    """Export, checker-validate, and return a compact ONNX operator manifest."""
    model = load_converted_model_from_qat(checkpoint)
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)

    # The legacy exporter preserves quantized PyTorch modules as ONNX Q/DQ.
    # The model is deliberately static: BudgetCUDA has fixed tensor geometry.
    torch.onnx.export(
        model,
        torch.rand(1, 3, 64, 64),
        output,
        opset_version=13,
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,
    )

    onnx_model = onnx.load(output)
    onnx.checker.check_model(onnx_model)
    operators = collections.Counter(node.op_type for node in onnx_model.graph.node)
    sample = np.random.default_rng(42).random((1, 3, 64, 64), dtype=np.float32)
    torch_logits = model(torch.from_numpy(sample)).detach().cpu().numpy()
    try:
        import onnxruntime as ort

        onnx_logits = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, {"input": sample})[0]
        max_abs_error: float | None = float(np.max(np.abs(torch_logits - onnx_logits)))
    except ImportError:
        max_abs_error = None
    manifest = {
        "checkpoint": str(checkpoint),
        "onnx": str(output),
        "opset": int(onnx_model.opset_import[0].version),
        "operators": dict(sorted(operators.items())),
        "input": {"name": "input", "shape": [1, 3, 64, 64], "dtype": "float32"},
        "output": {"name": "logits", "shape": [1, 3], "dtype": "float32"},
        "quantization": "PyTorch QAT Q/DQ INT8",
        "onnxruntime_max_abs_error": max_abs_error,
        "budgetcuda_import": "not yet supported: requires Q/DQ lowering to signed INT8 QLinearConv",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=QAT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(export_qat_int8_onnx(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
