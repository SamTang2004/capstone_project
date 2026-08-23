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
from .compiler import GraphBuilder, ResidentBufferPool, compose_model_packages
from .diagnostics import RunDiagnostics, summarize_cycle_report
from .dtypes import BufferUsage, DType
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
    diagnostics: RunDiagnostics | None = None


class DeiTManifestRunner:
    """Load a complete DeiT into DDR and execute it with one graph command."""

    def __init__(self, device, manifest_path: str | Path):
        self.device = device
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self._packages = {}
        self._resident = ResidentBufferPool(device)
        self._full_package = None
        self._full_model = None
        self._full_layer_names: tuple[str, ...] = ()
        self._full_layer_stages: tuple[str, ...] = ()
        self._closed = False
        self._validate_manifest()

    @property
    def resident_weight_bytes(self) -> int:
        if self._full_package is not None:
            return sum(
                buffer.size_bytes for buffer in self._full_package.buffers
                if buffer.usage == BufferUsage.WEIGHT)
        return self._resident.bytes

    @property
    def resident_buffer_count(self) -> int:
        if self._full_package is not None:
            return sum(
                buffer.usage == BufferUsage.WEIGHT
                for buffer in self._full_package.buffers)
        return len(self._resident.tensors)

    @property
    def resident_total_bytes(self) -> int:
        if self._full_package is None:
            return self._resident.bytes
        return (sum(buffer.size_bytes for buffer in self._full_package.buffers) +
                self.resident_descriptor_bytes)

    @property
    def resident_descriptor_bytes(self) -> int:
        if self._full_package is None:
            return 0
        return len(self._full_package.serialize_descriptor())

    @property
    def resident_total_buffer_count(self) -> int:
        if self._full_package is None:
            return len(self._resident.tensors)
        return len(self._full_package.buffers) + 1  # Model descriptor allocation.

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._full_model is not None:
            self._full_model.close()
            self._full_model = None
        self._resident.close()

    def _cache_package(self, name: str, build):
        if self._closed:
            raise RuntimeError("DeiT runner is closed")
        package = self._packages.get(name)
        if package is None:
            package = build()
            self._packages[name] = package
        return package

    def _attention_package(self, block_index: int):
        def build():
            block = self.manifest["blocks"][block_index]
            base = f"blocks.{block_index}"
            q, q_meta = self._linear(f"{base}.attn.q")
            k, k_meta = self._linear(f"{base}.attn.k")
            v, v_meta = self._linear(f"{base}.attn.v")
            projection, projection_meta = self._linear(f"{base}.attn.proj")
            hidden = int(self.manifest["hidden_dim"])
            heads = int(self.manifest["heads"])
            q_scale = float(q_meta["output_scale"])
            projection_input_scale = float(projection_meta["input_scale"])
            return build_deit_attention_block_package(
                int(self.manifest["tokens"]), hidden, heads, q, k, v, projection,
                gamma_q8=_q8(block["norm1_gamma"]), beta_q8=_q8(block["norm1_beta"]),
                q_bias_int32=q_meta.get("bias_int32"), k_bias_int32=k_meta.get("bias_int32"),
                v_bias_int32=v_meta.get("bias_int32"),
                output_bias_int32=projection_meta.get("bias_int32"),
                qkv_requant=tuple(q_meta["requant"]),
                score_requant=_fixed_multiplier(
                    (q_scale * q_scale) / max(1.0, (hidden // heads) ** 0.5)),
                context_requant=_fixed_multiplier(
                    (q_scale / 127.0) / projection_input_scale),
                output_requant=tuple(projection_meta["requant"]),
                weight_int4=int(self.manifest["weight_bits"]) == 4,
                name=f"deit_block_{block_index}_attention",
            )
        return self._cache_package(f"block_{block_index}.attention", build)

    def _mlp_package(self, block_index: int):
        def build():
            block = self.manifest["blocks"][block_index]
            base = f"blocks.{block_index}"
            fc1, fc1_meta = self._linear(f"{base}.mlp.fc1")
            fc2, fc2_meta = self._linear(f"{base}.mlp.fc2")
            return build_deit_mlp_block_package(
                int(self.manifest["tokens"]), int(self.manifest["hidden_dim"]),
                int(self.manifest["mlp_dim"]), fc1, fc2,
                gamma_q8=_q8(block["norm2_gamma"]), beta_q8=_q8(block["norm2_beta"]),
                fc1_bias_int32=fc1_meta.get("bias_int32"),
                fc2_bias_int32=fc2_meta.get("bias_int32"),
                fc1_requant=tuple(fc1_meta["requant"]),
                fc2_requant=tuple(fc2_meta["requant"]), accelerator_layout=True,
                weight_int4=int(self.manifest["weight_bits"]) == 4,
                name=f"deit_block_{block_index}_mlp",
            )
        return self._cache_package(f"block_{block_index}.mlp", build)

    def _final_norm_package(self):
        return self._cache_package("final_layernorm", lambda: build_deit_layernorm_package(
            int(self.manifest["tokens"]), int(self.manifest["hidden_dim"]),
            _q8(self.manifest["final_norm"]["gamma"]),
            _q8(self.manifest["final_norm"]["beta"]),
        ))

    def _head_package(self):
        def build():
            head, metadata = self._linear("head")
            return build_deit_linear_package(
                1, int(self.manifest["hidden_dim"]), len(self.manifest["classes"]), head,
                bias_int32=metadata.get("bias_int32"), requant=tuple(metadata["requant"]),
                weight_int4=int(self.manifest["weight_bits"]) == 4, name="deit_head",
            )
        return self._cache_package("six_class_head", build)

    def _patch_package(self):
        def build():
            patch = int(self.manifest["patch_size"])
            weights, metadata = self._linear("patch_embed.linear")
            return build_deit_linear_package(
                (224 // patch) ** 2, 3 * patch * patch, int(self.manifest["hidden_dim"]), weights,
                bias_int32=metadata.get("bias_int32"), requant=tuple(metadata["requant"]),
                weight_int4=int(self.manifest["weight_bits"]) == 4,
                name="deit_patch_embedding",
            )
        return self._cache_package("patch_embedding", build)

    def _token_embedding_package(self):
        def build():
            rows = int(self.manifest["tokens"])
            hidden = int(self.manifest["hidden_dim"])
            metadata = self.manifest["class_position_tokens"]
            patch_scale = float(self.manifest["tensors"]["patch_embed.linear"]["output_scale"])
            position_scale = float(metadata["scale"])
            token_scale = max(patch_scale, position_scale)
            raw = (self.root / metadata["file"]).read_bytes()
            expected = rows * hidden
            if len(raw) != expected:
                raise ValueError(f"Class/position tensor has {len(raw)} bytes, expected {expected}")
            position = []
            for offset in range(0, expected, hidden):
                row = []
                for value in raw[offset:offset + hidden]:
                    signed = value if value < 128 else value - 256
                    row.append(max(-128, min(127, int(round(signed * position_scale / token_scale)))))
                position.append(row)
            graph = GraphBuilder("deit_token_embedding")
            inp = graph.add_buffer("input", (rows - 1, hidden), DType.INT8)
            constant = graph.add_buffer(
                "class_position", (rows, hidden), DType.INT8,
                BufferUsage.WEIGHT, position)
            scale_ratio = patch_scale / token_scale
            scale_lut = [
                max(-128, min(127, int(round(value * scale_ratio))))
                for value in range(-128, 128)
            ]
            scale_lut = scale_lut[128:] + scale_lut[:128]
            scale = graph.add_buffer(
                "patch_scale_lut", (256,), DType.INT8,
                BufferUsage.WEIGHT, scale_lut)
            output = graph.add_buffer(
                "output", (rows, hidden), DType.INT8, BufferUsage.OUTPUT)
            graph.token_embedding(
                inp, constant, scale, output, patch_rows=rows - 1, features=hidden)
            return graph.package(inp, output)
        return self._cache_package("token_embedding", build)

    def load(self, *, include_patch: bool = True):
        """Upload the full graph, constants, and workspaces to DDR once."""
        if not include_patch:
            return self._load_streaming_encoder()
        if self._full_model is not None:
            return self
        named_sequence = [
            ("patch_embedding", self._patch_package()),
            ("token_embedding", self._token_embedding_package()),
        ]
        for block_index in range(int(self.manifest["depth"])):
            named_sequence.extend((
                (f"block_{block_index}.attention", self._attention_package(block_index)),
                (f"block_{block_index}.mlp", self._mlp_package(block_index)),
            ))
        named_sequence.extend((
            ("final_layernorm", self._final_norm_package()),
            ("six_class_head", self._head_package()),
        ))
        sequence = [package for _, package in named_sequence]
        self._full_layer_names = tuple(
            f"{stage}.layer_{index}.{layer.op_type.name}"
            for stage, package in named_sequence
            for index, layer in enumerate(package.layers)
        )
        self._full_layer_stages = tuple(
            stage for stage, package in named_sequence for _ in package.layers)
        self._full_package = compose_model_packages("deit_full_model", sequence)
        self._full_model = self.device.load_model(self._full_package)
        return self

    def _load_streaming_encoder(self):
        packages = []
        for block_index in range(int(self.manifest["depth"])):
            packages.extend((self._attention_package(block_index), self._mlp_package(block_index)))
        packages.extend((self._final_norm_package(), self._head_package()))
        self._resident.upload(
            buffer for package in packages for buffer in package.buffers
            if buffer.usage == BufferUsage.WEIGHT)
        return self

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
        model = self.device.load_model(package, external_tensors=self._resident.tensors)
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
        self.load(include_patch=False)
        started = perf_counter()
        stage_times: list[tuple[str, float]] = []
        token_count = int(self.manifest["tokens"])
        hidden = int(self.manifest["hidden_dim"])
        values = _flat_int8(tokens)
        if len(values) != token_count * hidden:
            raise ValueError(f"Expected {token_count * hidden} token values")

        for block_index in range(int(self.manifest["depth"])):
            package = self._attention_package(block_index)
            stage_started = perf_counter()
            values = self._execute(package, values, stage=f"block_{block_index}.attention")
            stage_times.append((f"block_{block_index}.attention", perf_counter() - stage_started))

            package = self._mlp_package(block_index)
            stage_started = perf_counter()
            values = self._execute(package, values, stage=f"block_{block_index}.mlp")
            stage_times.append((f"block_{block_index}.mlp", perf_counter() - stage_started))

        package = self._final_norm_package()
        stage_started = perf_counter()
        values = self._execute(
            package, values, tiled_output=True, stage="final_layernorm")
        stage_times.append(("final_layernorm", perf_counter() - stage_started))

        package = self._head_package()
        stage_started = perf_counter()
        logits = self._execute(package, values[:hidden], tiled_input=True, stage="six_class_head")
        stage_times.append(("six_class_head", perf_counter() - stage_started))
        class_index = max(range(len(logits)), key=logits.__getitem__)
        return DeiTInferenceResult(
            class_index, self.manifest["classes"][class_index], tuple(logits),
            perf_counter() - started, tuple(stage_times),
        )

    def run_image(self, image, *, diagnostics: bool = False) -> DeiTInferenceResult:
        """Preprocess a PIL image/path, execute patch embedding, then all blocks."""
        self.load()
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

        metadata = self.manifest["tensors"]["patch_embed.linear"]
        input_scale = float(metadata["input_scale"])
        quantized = np.clip(np.rint(patches / input_scale), -128, 127).astype(np.int8)
        payload = pack_transformer_activations_int8(
            quantized.tolist(), quantized.shape[0], quantized.shape[1])
        graph_started = perf_counter()
        try:
            execution = self._full_model.run(payload, diagnostics=diagnostics or None)
            report = execution.diagnostics if diagnostics else None
            logits = _flat_int8(execution.output if diagnostics else execution)
        except Exception as exc:
            raise RuntimeError("Full resident DeiT graph failed") from exc
        graph_seconds = perf_counter() - graph_started
        class_index = max(range(len(logits)), key=logits.__getitem__)
        return DeiTInferenceResult(
            class_index,
            self.manifest["classes"][class_index],
            tuple(logits),
            perf_counter() - started,
            (("full_device_graph", graph_seconds),),
            report,
        )

    def cycle_report(self, diagnostics: RunDiagnostics) -> dict:
        """Aggregate a diagnostic run using the fused model's stage metadata."""
        return summarize_cycle_report(
            diagnostics, self._full_layer_names, self._full_layer_stages)
