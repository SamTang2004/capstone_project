from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .deit import (
    build_deit_attention_block_package,
    build_deit_layernorm_package,
    build_deit_linear_package,
    build_deit_mlp_block_package,
)
from .packing import (
    pack_transformer_activations_int8,
    unpack_transformer_activations_int8,
    unpack_transformer_weights,
)


def _fixed_multiplier(real_scale: float) -> tuple[int, int, int, int]:
    shift = 30
    multiplier = int(round(float(real_scale) * (1 << shift)))
    while abs(multiplier) > 0x7FFFFFFF and shift:
        shift -= 1
        multiplier = int(round(float(real_scale) * (1 << shift)))
    if abs(multiplier) > 0x7FFFFFFF:
        raise ValueError(f"Scale {real_scale} cannot be represented")
    return multiplier, shift, 0, 0


def _q8(values) -> list[int]:
    return [max(-32768, min(32767, int(round(float(value) * 256)))) for value in values]


def _flat_int8(result) -> list[int]:
    if hasattr(result, "reshape"):
        return [int(value) for value in result.reshape(-1).tolist()]
    return [int(value) for value in result]


@dataclass(frozen=True)
class DeiTInferenceResult:
    class_index: int
    class_name: str
    logits: tuple[int, ...]
    elapsed_seconds: float
    stage_seconds: tuple[tuple[str, float], ...]


class DeiTManifestRunner:
    """Stream a complete exported DeiT through the graph accelerator.

    One attention or MLP graph is resident at a time. This keeps the current
    128-buffer firmware ABI and 16/64 MiB CMA arenas usable for all 12 blocks.
    """

    def __init__(self, device, manifest_path: str | Path):
        self.device = device
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest()

    def _validate_manifest(self) -> None:
        manifest = self.manifest
        if manifest.get("version") != 1 or manifest.get("weight_bits") not in (4, 8):
            raise ValueError("Unsupported BudgetCUDA DeiT manifest")
        required = ("classes", "tokens", "hidden_dim", "heads", "depth", "mlp_dim", "tensors", "blocks")
        if any(key not in manifest for key in required):
            raise ValueError("Incomplete BudgetCUDA DeiT manifest")
        if len(manifest["blocks"]) != int(manifest["depth"]):
            raise ValueError("Manifest block count does not match depth")
        for name, tensor in manifest["tensors"].items():
            path = self.root / tensor["file"]
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != tensor["sha256"]:
                raise ValueError(f"Weight hash mismatch: {name}")

    def _linear(self, name: str) -> tuple[list[list[int]], dict]:
        tensor = self.manifest["tensors"][name]
        k, n = (int(value) for value in tensor["shape_k_n"])
        raw = (self.root / tensor["file"]).read_bytes()
        weights = unpack_transformer_weights(
            raw, k, n, int4=int(self.manifest["weight_bits"]) == 4)
        return weights, tensor

    def _execute(
        self, package, values, *, tiled_input: bool = False,
        tiled_output: bool = False, stage: str | None = None,
    ):
        model = self.device.load_model(package)
        try:
            payload = values
            if tiled_input:
                rows, columns = package.buffers[package.input_index].shape
                matrix = [values[row * columns:(row + 1) * columns] for row in range(rows)]
                payload = pack_transformer_activations_int8(matrix, rows, columns)
            try:
                result = model.run(payload)
                if tiled_output:
                    rows, columns = package.buffers[package.output_index].shape
                    flattened = result.reshape(-1) if hasattr(result, "reshape") else result
                    raw = bytes(int(value) & 0xFF for value in flattened)
                    unpacked = unpack_transformer_activations_int8(raw, rows, columns)
                    return [value for row in unpacked for value in row]
                return _flat_int8(result)
            except Exception as exc:
                if stage is None:
                    raise
                raise RuntimeError(f"DeiT stage {stage!r} failed") from exc
        finally:
            model.close()

    def run_tokens(self, tokens) -> DeiTInferenceResult:
        """Run pre-quantized row-major token values through all encoder blocks."""
        started = perf_counter()
        stage_times: list[tuple[str, float]] = []
        token_count = int(self.manifest["tokens"])
        hidden = int(self.manifest["hidden_dim"])
        heads = int(self.manifest["heads"])
        mlp = int(self.manifest["mlp_dim"])
        values = _flat_int8(tokens)
        if len(values) != token_count * hidden:
            raise ValueError(f"Expected {token_count * hidden} token values")
        int4 = int(self.manifest["weight_bits"]) == 4

        for block_index, block in enumerate(self.manifest["blocks"]):
            base = f"blocks.{block_index}"
            q, q_meta = self._linear(f"{base}.attn.q")
            k, k_meta = self._linear(f"{base}.attn.k")
            v, v_meta = self._linear(f"{base}.attn.v")
            projection, projection_meta = self._linear(f"{base}.attn.proj")
            q_scale = float(q_meta["output_scale"])
            projection_input_scale = float(projection_meta["input_scale"])
            score_requant = _fixed_multiplier((q_scale * q_scale) / max(1.0, (hidden // heads) ** 0.5))
            context_requant = _fixed_multiplier((q_scale / 127.0) / projection_input_scale)
            package = build_deit_attention_block_package(
                token_count, hidden, heads, q, k, v, projection,
                gamma_q8=_q8(block["norm1_gamma"]), beta_q8=_q8(block["norm1_beta"]),
                q_bias_int32=q_meta.get("bias_int32"), k_bias_int32=k_meta.get("bias_int32"),
                v_bias_int32=v_meta.get("bias_int32"),
                output_bias_int32=projection_meta.get("bias_int32"),
                qkv_requant=tuple(q_meta["requant"]), score_requant=score_requant,
                context_requant=context_requant, output_requant=tuple(projection_meta["requant"]),
                weight_int4=int4, name=f"deit_block_{block_index}_attention",
            )
            stage_started = perf_counter()
            values = self._execute(package, values, stage=f"block_{block_index}.attention")
            stage_times.append((f"block_{block_index}.attention", perf_counter() - stage_started))

            fc1, fc1_meta = self._linear(f"{base}.mlp.fc1")
            fc2, fc2_meta = self._linear(f"{base}.mlp.fc2")
            package = build_deit_mlp_block_package(
                token_count, hidden, mlp, fc1, fc2,
                gamma_q8=_q8(block["norm2_gamma"]), beta_q8=_q8(block["norm2_beta"]),
                fc1_bias_int32=fc1_meta.get("bias_int32"), fc2_bias_int32=fc2_meta.get("bias_int32"),
                fc1_requant=tuple(fc1_meta["requant"]), fc2_requant=tuple(fc2_meta["requant"]),
                accelerator_layout=True, weight_int4=int4, name=f"deit_block_{block_index}_mlp",
            )
            stage_started = perf_counter()
            values = self._execute(package, values, stage=f"block_{block_index}.mlp")
            stage_times.append((f"block_{block_index}.mlp", perf_counter() - stage_started))

        final_norm = self.manifest["final_norm"]
        package = build_deit_layernorm_package(
            token_count, hidden, _q8(final_norm["gamma"]), _q8(final_norm["beta"]),
        )
        stage_started = perf_counter()
        values = self._execute(
            package, values, tiled_output=True, stage="final_layernorm")
        stage_times.append(("final_layernorm", perf_counter() - stage_started))

        head, head_meta = self._linear("head")
        package = build_deit_linear_package(
            1, hidden, len(self.manifest["classes"]), head,
            bias_int32=head_meta.get("bias_int32"), requant=tuple(head_meta["requant"]),
            weight_int4=int4, name="deit_head",
        )
        stage_started = perf_counter()
        logits = self._execute(package, values[:hidden], tiled_input=True, stage="six_class_head")
        stage_times.append(("six_class_head", perf_counter() - stage_started))
        class_index = max(range(len(logits)), key=logits.__getitem__)
        return DeiTInferenceResult(
            class_index, self.manifest["classes"][class_index], tuple(logits),
            perf_counter() - started, tuple(stage_times),
        )

    def run_image(self, image) -> DeiTInferenceResult:
        """Preprocess a PIL image/path, execute patch embedding, then all blocks."""
        started = perf_counter()
        try:
            import numpy as np
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise RuntimeError("run_image requires numpy and Pillow") from exc
        image = Image.open(image) if isinstance(image, (str, Path)) else image
        image = image.convert("RGB")
        width, height = image.size
        scale = 248.0 / min(width, height)
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
        left, top = (image.width - 224) // 2, (image.height - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))
        array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)[:, None, None]
        std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)[:, None, None]
        array = (array - mean) / std
        patch = int(self.manifest["patch_size"])
        grid = 224 // patch
        patches = array.reshape(3, grid, patch, grid, patch).transpose(1, 3, 0, 2, 4).reshape(grid * grid, -1)

        weights, metadata = self._linear("patch_embed.linear")
        input_scale = float(metadata["input_scale"])
        quantized = np.clip(np.rint(patches / input_scale), -128, 127).astype(np.int8).reshape(-1).tolist()
        package = build_deit_linear_package(
            grid * grid, patches.shape[1], int(self.manifest["hidden_dim"]), weights,
            bias_int32=metadata.get("bias_int32"), requant=tuple(metadata["requant"]),
            weight_int4=int(self.manifest["weight_bits"]) == 4, name="deit_patch_embedding",
        )
        patch_started = perf_counter()
        patch_values = self._execute(package, quantized, tiled_input=True, stage="patch_embedding")
        patch_seconds = perf_counter() - patch_started

        position_meta = self.manifest["class_position_tokens"]
        position_raw = np.frombuffer((self.root / position_meta["file"]).read_bytes(), dtype=np.int8).astype(np.int32)
        patch_scale = float(metadata["output_scale"])
        position_scale = float(position_meta["scale"])
        token_scale = max(patch_scale, position_scale)
        patch_matrix = np.asarray(patch_values, dtype=np.int32).reshape(grid * grid, -1)
        patch_matrix = np.rint(patch_matrix * (patch_scale / token_scale)).astype(np.int32)
        positions = np.rint(position_raw * (position_scale / token_scale)).astype(np.int32).reshape(grid * grid + 1, -1)
        tokens = positions
        tokens[1:] += patch_matrix
        tokens = np.clip(tokens, -128, 127).astype(np.int8)
        result = self.run_tokens(tokens)
        return DeiTInferenceResult(
            result.class_index,
            result.class_name,
            result.logits,
            perf_counter() - started,
            (("patch_embedding", patch_seconds), *result.stage_seconds),
        )
