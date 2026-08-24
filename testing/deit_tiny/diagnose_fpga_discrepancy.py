"""Capture per-layer FPGA outputs to locate overlay discrepancies.

The complete model and all of its buffers remain resident in DDR.  Each probe
uploads only a short-lived descriptor containing a prefix of the full graph,
then reads the final buffer of that prefix.  This makes every capture start
from the same image input without introducing host/device transfers between
model layers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "FPGA"))

from budgetcuda import DeiTManifestRunner, Device
from budgetcuda.compiler import ModelPackage
from budgetcuda.packing import pack_transformer_activations_int8


def _image_payload(image_path: Path, runner: DeiTManifestRunner) -> bytes:
    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("diagnosis requires numpy and Pillow") from exc

    image = Image.open(image_path).convert("RGB")
    scale = 248.0 / min(image.size)
    image = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        Image.Resampling.BICUBIC,
    )
    left, top = (image.width - 224) // 2, (image.height - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
    array = (array - mean) / std
    patch = int(runner.manifest["patch_size"])
    grid = 224 // patch
    patches = array.reshape(3, grid, patch, grid, patch).transpose(
        1, 3, 0, 2, 4).reshape(grid * grid, -1)
    input_scale = float(runner.manifest["tensors"]["patch_embed.linear"]["input_scale"])
    quantized = np.clip(np.rint(patches / input_scale), -128, 127).astype(np.int8)
    return pack_transformer_activations_int8(
        quantized.tolist(), quantized.shape[0], quantized.shape[1])


def capture(
    runner: DeiTManifestRunner,
    payload: bytes,
    layer_indices: list[int],
    raw_dir: Path | None = None,
) -> list[dict]:
    runner.load()
    package = runner._full_package
    model = runner._full_model
    if package is None or model is None:
        raise RuntimeError("Full resident model failed to load")
    if len(model.tensors) != len(package.buffers):
        raise RuntimeError("Model tensor table does not match its package")

    runner.device.write(model.input_tensor, payload)
    buffer_ids = [tensor.buffer_id for tensor in model.tensors]
    captures = []
    for layer_index in layer_indices:
        layer = package.layers[layer_index]
        prefix = ModelPackage(
            package.buffers,
            package.layers[:layer_index + 1],
            package.input_index,
            layer.output,
            flags=package.flags,
        )
        descriptor = prefix.serialize_descriptor(buffer_ids)
        descriptor_tensor = runner.device.alloc(
            (len(descriptor),), "int8", usage="descriptor")
        try:
            runner.device.write(descriptor_tensor, descriptor)
            output_tensor = model.tensors[layer.output]
            runner.device.execute_graph(
                descriptor_tensor, model.input_tensor, output_tensor)
            raw = runner.device.protocol.read(
                output_tensor.buffer_id, 0, output_tensor.size_bytes)
        finally:
            runner.device.free(descriptor_tensor)
        if raw_dir is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / f"layer_{layer_index:03d}.bin").write_bytes(raw)
        captures.append({
            "layer_index": layer_index,
            "layer_name": runner._full_layer_names[layer_index],
            "stage": runner._full_layer_stages[layer_index],
            "op_type": layer.op_type.name,
            "output_buffer": package.buffers[layer.output].name,
            "output_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "first_32_bytes_hex": raw[:32].hex(),
        })
    return captures


def _parse_layers(specification: str, layer_count: int) -> list[int]:
    if specification == "all":
        return list(range(layer_count))
    indices = sorted({int(value) for value in specification.split(",")})
    if not indices or indices[0] < 0 or indices[-1] >= layer_count:
        raise ValueError(f"Layer indices must be in 0..{layer_count - 1}")
    return indices


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--manifest", type=Path,
                        default=here / "artifacts" / "budgetcuda_w8a8" / "manifest.json")
    parser.add_argument("--address", default="192.168.2.99")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--layers", default="all",
                        help="Comma-separated zero-based layer indices, or 'all'")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path,
                        help="Optionally save each captured layer as raw bytes")
    args = parser.parse_args()

    with DeiTManifestRunner(
            Device(args.address, args.port, args.timeout), args.manifest) as runner:
        runner.load()
        layer_count = len(runner._full_package.layers)
        layers = _parse_layers(args.layers, layer_count)
        payload = _image_payload(args.image, runner)
        records = capture(runner, payload, layers, args.raw_dir)
    report = {
        "image": str(args.image.resolve()),
        "manifest": str(args.manifest.resolve()),
        "layer_count": layer_count,
        "captured_layer_count": len(records),
        "layers": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
