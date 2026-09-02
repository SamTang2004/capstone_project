from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


_CONFIG_STRUCT = struct.Struct("<7i")
_TOKENIZER_HEADER = struct.Struct("<I")
_TOKENIZER_ENTRY = struct.Struct("<fI")
_BYTE_TOKEN = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


@dataclass(frozen=True)
class StoriesConfig:
    dim: int
    hidden_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    seq_len: int
    shared_classifier: bool = True

    def __post_init__(self) -> None:
        values = {
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
        }
        for name, value in values.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.dim % self.n_heads:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.vocab_size > 10_000_000 or self.dim > 1_000_000:
            raise ValueError("checkpoint dimensions are implausibly large")

    @property
    def head_size(self) -> int:
        return self.dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.head_size * self.n_kv_heads

    def require_stories15m(self) -> None:
        expected = {
            "dim": 288,
            "hidden_dim": 768,
            "n_layers": 6,
            "n_heads": 6,
            "n_kv_heads": 6,
            "vocab_size": 32_000,
            "seq_len": 256,
        }
        mismatches = [
            f"{name}={getattr(self, name)} (expected {value})"
            for name, value in expected.items()
            if getattr(self, name) != value
        ]
        if mismatches:
            raise ValueError("not the stories15M model: " + ", ".join(mismatches))

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "vocab_size": self.vocab_size,
            "seq_len": self.seq_len,
            "head_size": self.head_size,
            "kv_dim": self.kv_dim,
            "shared_classifier": self.shared_classifier,
        }


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shape: tuple[int, ...]
    offset_bytes: int
    size_bytes: int


class Llama2CCheckpoint:
    """Read the legacy llama2.c float checkpoint used by stories15M.

    Tensor views are backed by a read-only NumPy memory map. The reader checks
    the complete file size, including the legacy RoPE tables that llama2.c
    retains in the file but no longer consumes.
    """

    def __init__(self, path: str | Path, *, require_stories15m: bool = False):
        self.path = Path(path)
        try:
            file_size = self.path.stat().st_size
        except FileNotFoundError:
            raise FileNotFoundError(f"checkpoint not found: {self.path}") from None
        if file_size < _CONFIG_STRUCT.size:
            raise ValueError("checkpoint is smaller than the 28-byte llama2.c header")

        with self.path.open("rb") as source:
            header = source.read(_CONFIG_STRUCT.size)
        dim, hidden, layers, heads, kv_heads, signed_vocab, seq_len = _CONFIG_STRUCT.unpack(header)
        if signed_vocab == 0:
            raise ValueError("vocab_size must not be zero")
        self.config = StoriesConfig(
            dim, hidden, layers, heads, kv_heads, abs(signed_vocab), seq_len,
            shared_classifier=signed_vocab > 0,
        )
        if require_stories15m:
            self.config.require_stories15m()

        expected_bytes = _CONFIG_STRUCT.size + self._weight_float_count() * 4
        if file_size != expected_bytes:
            relation = "truncated" if file_size < expected_bytes else "has trailing data"
            raise ValueError(
                f"checkpoint {relation}: expected {expected_bytes} bytes from the header, "
                f"found {file_size}"
            )

        self._mapped = np.memmap(
            self.path, dtype="<f4", mode="r", offset=_CONFIG_STRUCT.size,
            shape=(self._weight_float_count(),),
        )
        self._records: list[TensorRecord] = []
        cursor = 0

        def take(name: str, shape: tuple[int, ...]) -> np.ndarray:
            nonlocal cursor
            count = math.prod(shape)
            start = cursor
            cursor += count
            self._records.append(TensorRecord(
                name, shape, _CONFIG_STRUCT.size + start * 4, count * 4,
            ))
            return self._mapped[start:cursor].reshape(shape)

        c = self.config
        self.token_embedding_table = take("token_embedding_table", (c.vocab_size, c.dim))
        self.rms_att_weight = take("rms_att_weight", (c.n_layers, c.dim))
        self.wq = take("wq", (c.n_layers, c.dim, c.dim))
        self.wk = take("wk", (c.n_layers, c.kv_dim, c.dim))
        self.wv = take("wv", (c.n_layers, c.kv_dim, c.dim))
        self.wo = take("wo", (c.n_layers, c.dim, c.dim))
        self.rms_ffn_weight = take("rms_ffn_weight", (c.n_layers, c.dim))
        self.w1 = take("w1", (c.n_layers, c.hidden_dim, c.dim))
        self.w2 = take("w2", (c.n_layers, c.dim, c.hidden_dim))
        self.w3 = take("w3", (c.n_layers, c.hidden_dim, c.dim))
        self.rms_final_weight = take("rms_final_weight", (c.dim,))
        rope_shape = (c.seq_len, c.head_size // 2)
        self.legacy_rope_real = take("legacy_rope_real", rope_shape)
        self.legacy_rope_imag = take("legacy_rope_imag", rope_shape)
        if c.shared_classifier:
            self.wcls = self.token_embedding_table
        else:
            self.wcls = take("wcls", (c.vocab_size, c.dim))
        if cursor != self._mapped.size:
            raise AssertionError("internal checkpoint layout did not consume the mapped file")

    def _weight_float_count(self) -> int:
        c = self.config
        count = c.vocab_size * c.dim
        count += c.n_layers * c.dim
        count += c.n_layers * c.dim * c.dim
        count += c.n_layers * c.kv_dim * c.dim * 2
        count += c.n_layers * c.dim * c.dim
        count += c.n_layers * c.dim
        count += c.n_layers * c.hidden_dim * c.dim * 3
        count += c.dim
        count += c.seq_len * c.head_size
        if not c.shared_classifier:
            count += c.vocab_size * c.dim
        return count

    @property
    def tensor_records(self) -> tuple[TensorRecord, ...]:
        return tuple(self._records)

    @property
    def sha256(self) -> str:
        digest = hashlib.sha256()
        with self.path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def ledger(self) -> dict:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "file_size_bytes": self.path.stat().st_size,
            "config": self.config.as_dict(),
            "tensors": [
                {
                    "name": item.name,
                    "shape": list(item.shape),
                    "offset_bytes": item.offset_bytes,
                    "size_bytes": item.size_bytes,
                }
                for item in self.tensor_records
            ],
        }


class Llama2CTokenizer:
    """Tokenizer compatible with llama2.c's tokenizer.bin BPE format."""

    def __init__(self, path: str | Path, vocab_size: int):
        self.path = Path(path)
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        with self.path.open("rb") as source:
            header = source.read(_TOKENIZER_HEADER.size)
            if len(header) != _TOKENIZER_HEADER.size:
                raise ValueError("tokenizer is missing max_token_length")
            (self.max_token_length,) = _TOKENIZER_HEADER.unpack(header)
            pieces: list[str] = []
            scores: list[float] = []
            for index in range(vocab_size):
                entry = source.read(_TOKENIZER_ENTRY.size)
                if len(entry) != _TOKENIZER_ENTRY.size:
                    raise ValueError(f"tokenizer ended before entry {index}")
                score, length = _TOKENIZER_ENTRY.unpack(entry)
                if length > self.max_token_length:
                    raise ValueError(
                        f"tokenizer entry {index} length {length} exceeds max {self.max_token_length}"
                    )
                raw = source.read(length)
                if len(raw) != length:
                    raise ValueError(f"tokenizer entry {index} is truncated")
                try:
                    piece = raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"tokenizer entry {index} is not UTF-8") from exc
                pieces.append(piece)
                scores.append(float(score))
            if source.read(1):
                raise ValueError("tokenizer has trailing data")
        self.pieces = tuple(pieces)
        self.scores = np.asarray(scores, dtype=np.float32)
        self._piece_to_id = {piece: index for index, piece in enumerate(self.pieces)}

    def encode(self, text: str, *, bos: bool = True, eos: bool = False) -> list[int]:
        tokens: list[int] = [1] if bos else []
        if text:
            dummy_prefix = self._piece_to_id.get(" ")
            if dummy_prefix is not None:
                tokens.append(dummy_prefix)
        for character in text:
            token = self._piece_to_id.get(character)
            if token is not None:
                tokens.append(token)
                continue
            for byte in character.encode("utf-8"):
                fallback = byte + 3
                if fallback >= len(self.pieces):
                    raise ValueError(
                        f"tokenizer has no piece for {character!r} and lacks byte fallback id {fallback}"
                    )
                tokens.append(fallback)

        while len(tokens) >= 2:
            best_score = -math.inf
            best_index = -1
            best_id = -1
            for index in range(len(tokens) - 1):
                merged = self.pieces[tokens[index]] + self.pieces[tokens[index + 1]]
                merged_id = self._piece_to_id.get(merged)
                if merged_id is not None and self.scores[merged_id] > best_score:
                    best_score = float(self.scores[merged_id])
                    best_index = index
                    best_id = merged_id
            if best_index < 0:
                break
            tokens[best_index:best_index + 2] = [best_id]
        if eos:
            tokens.append(2)
        return tokens

    def decode_piece(self, previous_token: int, token: int) -> bytes:
        if token < 0 or token >= len(self.pieces):
            raise ValueError(f"token id {token} is outside the vocabulary")
        if token in (1, 2):
            return b""
        piece = self.pieces[token]
        if previous_token == 1 and piece.startswith(" "):
            piece = piece[1:]
        byte_match = _BYTE_TOKEN.match(piece)
        if byte_match:
            return bytes([int(byte_match.group(1), 16)])
        return piece.encode("utf-8")

    def decode(self, tokens: Iterable[int]) -> str:
        previous = 1
        result = bytearray()
        for token in tokens:
            token = int(token)
            result.extend(self.decode_piece(previous, token))
            previous = token
        return result.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class StoriesForwardResult:
    token: int
    position: int
    logits: np.ndarray
    trace: Mapping[str, np.ndarray]


class StoriesFloatReference:
    """Deterministic FP32, single-sequence llama2.c reference with trace capture."""

    def __init__(self, checkpoint: Llama2CCheckpoint):
        self.checkpoint = checkpoint
        self.weights = checkpoint
        c = checkpoint.config
        self.key_cache = np.zeros((c.n_layers, c.seq_len, c.kv_dim), dtype=np.float32)
        self.value_cache = np.zeros_like(self.key_cache)
        self.next_position = 0

    def reset(self) -> None:
        self.key_cache.fill(0)
        self.value_cache.fill(0)
        self.next_position = 0

    @staticmethod
    def _copy(value) -> np.ndarray:
        return np.asarray(value, dtype=np.float32).copy()

    @staticmethod
    def _rmsnorm(value: np.ndarray, weight: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float32)
        square_sum = np.sum(value * value, dtype=np.float32)
        scale = np.float32(1.0) / np.sqrt(
            np.float32(square_sum / np.float32(value.size) + np.float32(1e-5)),
            dtype=np.float32,
        )
        return np.asarray(weight * (value * scale), dtype=np.float32)

    @staticmethod
    def _softmax(value: np.ndarray) -> np.ndarray:
        shifted = np.asarray(value - np.max(value), dtype=np.float32)
        exponent = np.exp(shifted, dtype=np.float32)
        return np.asarray(exponent / np.sum(exponent, dtype=np.float32), dtype=np.float32)

    @staticmethod
    def _matmul(weight: np.ndarray, value: np.ndarray) -> np.ndarray:
        return np.asarray(weight @ value, dtype=np.float32)

    def _activation(self, value: np.ndarray, _name: str) -> np.ndarray:
        return np.asarray(value, dtype=np.float32)

    def _probability(self, value: np.ndarray, _name: str) -> np.ndarray:
        return np.asarray(value, dtype=np.float32)

    def _rope(self, q: np.ndarray, k: np.ndarray, position: int) -> None:
        c = self.checkpoint.config
        for index in range(0, c.dim, 2):
            head_dim = index % c.head_size
            frequency = np.float32(1.0 / (10000.0 ** (head_dim / c.head_size)))
            angle = np.float32(position) * frequency
            cosine = np.float32(math.cos(float(angle)))
            sine = np.float32(math.sin(float(angle)))
            first, second = np.float32(q[index]), np.float32(q[index + 1])
            q[index] = first * cosine - second * sine
            q[index + 1] = first * sine + second * cosine
            if index < c.kv_dim:
                first, second = np.float32(k[index]), np.float32(k[index + 1])
                k[index] = first * cosine - second * sine
                k[index + 1] = first * sine + second * cosine

    def forward(self, token: int, position: int | None = None, *, capture: bool = True) -> StoriesForwardResult:
        c = self.checkpoint.config
        w = self.weights
        if position is None:
            position = self.next_position
        if position != self.next_position:
            raise ValueError(
                f"reference expects sequential position {self.next_position}, got {position}; call reset()"
            )
        if position >= c.seq_len:
            raise ValueError(f"position {position} exceeds seq_len {c.seq_len}")
        if token < 0 or token >= c.vocab_size:
            raise ValueError(f"token id {token} is outside vocabulary size {c.vocab_size}")

        trace: dict[str, np.ndarray] = {}
        x = self._activation(self._copy(w.token_embedding_table[token]), "embedding")
        if capture:
            trace["embedding"] = self._copy(x)

        for layer in range(c.n_layers):
            prefix = f"layer_{layer:02d}"
            rms_attention = self._activation(
                self._rmsnorm(x, w.rms_att_weight[layer]), f"{prefix}.rms_attention",
            )
            q = self._activation(
                self._matmul(w.wq[layer], rms_attention), f"{prefix}.q",
            )
            k = self._activation(
                self._matmul(w.wk[layer], rms_attention), f"{prefix}.k",
            )
            v = self._activation(
                self._matmul(w.wv[layer], rms_attention), f"{prefix}.v",
            )
            self._rope(q, k, position)
            q = self._activation(q, f"{prefix}.q_rope")
            k = self._activation(k, f"{prefix}.k_rope")
            self.key_cache[layer, position] = k
            self.value_cache[layer, position] = v

            attention_concat = np.empty(c.dim, dtype=np.float32)
            attention_logits = np.empty((c.n_heads, position + 1), dtype=np.float32)
            attention_probabilities = np.empty_like(attention_logits)
            kv_multiplier = c.n_heads // c.n_kv_heads
            for head in range(c.n_heads):
                q_start = head * c.head_size
                q_head = q[q_start:q_start + c.head_size]
                kv_head = head // kv_multiplier
                kv_start = kv_head * c.head_size
                keys = self.key_cache[layer, :position + 1, kv_start:kv_start + c.head_size]
                scores = self._activation(
                    np.asarray(
                        (keys @ q_head) / np.float32(math.sqrt(c.head_size)), dtype=np.float32,
                    ),
                    f"{prefix}.attention_logits",
                )
                probabilities = self._probability(
                    self._softmax(scores), f"{prefix}.attention_probabilities",
                )
                attention_logits[head] = scores
                attention_probabilities[head] = probabilities
                values = self.value_cache[layer, :position + 1, kv_start:kv_start + c.head_size]
                attention_concat[q_start:q_start + c.head_size] = self._activation(
                    np.asarray(probabilities @ values, dtype=np.float32),
                    f"{prefix}.attention_head",
                )
            attention_concat = self._activation(
                attention_concat, f"{prefix}.attention_concat",
            )
            attention_output = self._activation(
                self._matmul(w.wo[layer], attention_concat), f"{prefix}.attention_output",
            )
            x = self._activation(
                np.asarray(x + attention_output, dtype=np.float32), f"{prefix}.post_attention",
            )
            post_attention = self._copy(x)

            rms_ffn = self._activation(
                self._rmsnorm(x, w.rms_ffn_weight[layer]), f"{prefix}.rms_ffn",
            )
            gate = self._activation(
                self._matmul(w.w1[layer], rms_ffn), f"{prefix}.ffn_gate",
            )
            up = self._activation(
                self._matmul(w.w3[layer], rms_ffn), f"{prefix}.ffn_up",
            )
            silu_gate = self._activation(
                np.asarray(
                    gate * (np.float32(1.0) / (
                        np.float32(1.0) + np.exp(-gate, dtype=np.float32)
                    )),
                    dtype=np.float32,
                ),
                f"{prefix}.silu_gate",
            )
            swiglu = self._activation(
                np.asarray(silu_gate * up, dtype=np.float32), f"{prefix}.swiglu",
            )
            ffn_output = self._activation(
                self._matmul(w.w2[layer], swiglu), f"{prefix}.ffn_output",
            )
            x = self._activation(
                np.asarray(x + ffn_output, dtype=np.float32), f"{prefix}.output",
            )

            if capture:
                trace[f"{prefix}.rms_attention"] = self._copy(rms_attention)
                trace[f"{prefix}.q"] = self._copy(q)
                trace[f"{prefix}.k"] = self._copy(k)
                trace[f"{prefix}.v"] = self._copy(v)
                trace[f"{prefix}.key_cache"] = self._copy(
                    self.key_cache[layer, :position + 1]
                )
                trace[f"{prefix}.value_cache"] = self._copy(
                    self.value_cache[layer, :position + 1]
                )
                trace[f"{prefix}.attention_logits"] = self._copy(attention_logits)
                trace[f"{prefix}.attention_probabilities"] = self._copy(attention_probabilities)
                trace[f"{prefix}.attention_concat"] = self._copy(attention_concat)
                trace[f"{prefix}.attention_output"] = self._copy(attention_output)
                trace[f"{prefix}.post_attention"] = post_attention
                trace[f"{prefix}.rms_ffn"] = self._copy(rms_ffn)
                trace[f"{prefix}.ffn_gate"] = self._copy(gate)
                trace[f"{prefix}.ffn_up"] = self._copy(up)
                trace[f"{prefix}.swiglu"] = self._copy(swiglu)
                trace[f"{prefix}.ffn_output"] = self._copy(ffn_output)
                trace[f"{prefix}.output"] = self._copy(x)

        final_norm = self._activation(self._rmsnorm(x, w.rms_final_weight), "final_norm")
        logits = self._matmul(w.wcls, final_norm)
        if capture:
            trace["final_norm"] = self._copy(final_norm)
            trace["logits"] = self._copy(logits)
        self.next_position += 1
        return StoriesForwardResult(int(token), int(position), logits, trace)

    def run_tokens(self, tokens: Sequence[int], *, capture: bool = True) -> tuple[StoriesForwardResult, ...]:
        if not tokens:
            raise ValueError("at least one token is required")
        if len(tokens) > self.checkpoint.config.seq_len:
            raise ValueError("token sequence exceeds checkpoint seq_len")
        self.reset()
        return tuple(self.forward(int(token), capture=capture) for token in tokens)

    def generate_greedy(
        self, prompt_tokens: Sequence[int], max_new_tokens: int, *, capture: bool = True,
    ) -> tuple[tuple[int, ...], tuple[StoriesForwardResult, ...]]:
        if not prompt_tokens:
            raise ValueError("at least one prompt token is required")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if len(prompt_tokens) + max_new_tokens > self.checkpoint.config.seq_len:
            raise ValueError("prompt and generated tokens exceed checkpoint seq_len")
        tokens = [int(token) for token in prompt_tokens]
        self.reset()
        results: list[StoriesForwardResult] = []
        for position, token in enumerate(tokens[:-1]):
            results.append(self.forward(token, position, capture=capture))
        current = tokens[-1]
        for _ in range(max_new_tokens):
            result = self.forward(current, len(results), capture=capture)
            results.append(result)
            current = int(np.argmax(result.logits))
            tokens.append(current)
        return tuple(tokens), tuple(results)


def export_stories_golden(
    checkpoint: Llama2CCheckpoint,
    output_prefix: str | Path,
    prompt_tokens: Sequence[int],
    *,
    max_new_tokens: int = 0,
    top_k: int = 16,
    tokenizer: Llama2CTokenizer | None = None,
    reference: StoriesFloatReference | None = None,
    golden_format: str = "budgetcuda-stories-fp32-golden",
    extra_metadata: Mapping[str, object] | None = None,
) -> tuple[Path, Path]:
    """Export full per-layer traces and a reviewable JSON index.

    The NPZ contains every captured tensor, including the complete vocabulary
    logits for every evaluated position. JSON contains tensor hashes, shapes,
    top logits, token flow, model identity, and optional decoded text.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path = prefix.with_suffix(".npz")
    json_path = prefix.with_suffix(".json")
    if reference is None:
        reference = StoriesFloatReference(checkpoint)
    if reference.checkpoint.config != checkpoint.config:
        raise ValueError("golden reference and checkpoint configs must match")
    if max_new_tokens:
        output_tokens, results = reference.generate_greedy(
            prompt_tokens, max_new_tokens, capture=True,
        )
    else:
        output_tokens = tuple(int(token) for token in prompt_tokens)
        results = reference.run_tokens(output_tokens, capture=True)

    arrays: dict[str, np.ndarray] = {}
    step_metadata: list[dict] = []
    for step, result in enumerate(results):
        tensors: list[dict] = []
        for trace_name, value in result.trace.items():
            key = f"step_{step:03d}__{trace_name.replace('.', '__')}"
            array = np.ascontiguousarray(value, dtype=np.float32)
            arrays[key] = array
            tensors.append({
                "name": trace_name,
                "npz_key": key,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            })
        count = min(top_k, result.logits.size)
        indices = np.argsort(-result.logits, kind="stable")[:count]
        step_metadata.append({
            "step": step,
            "position": result.position,
            "input_token": result.token,
            "actual_next_token": (
                int(output_tokens[step + 1]) if step + 1 < len(output_tokens) else None
            ),
            "greedy_next_token": int(np.argmax(result.logits)),
            "top_logits": [
                {"token": int(index), "logit": float(result.logits[index])}
                for index in indices
            ],
            "tensors": tensors,
        })

    np.savez_compressed(npz_path, **arrays)
    metadata = {
        "format": golden_format,
        "version": 1,
        "checkpoint": {
            "path": str(checkpoint.path),
            "sha256": checkpoint.sha256,
            "config": checkpoint.config.as_dict(),
        },
        "prompt_tokens": [int(token) for token in prompt_tokens],
        "max_new_tokens": int(max_new_tokens),
        "output_tokens": list(output_tokens),
        "decoded_text": tokenizer.decode(output_tokens) if tokenizer is not None else None,
        "npz_file": npz_path.name,
        "npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "steps": step_metadata,
    }
    if extra_metadata:
        overlap = set(metadata).intersection(extra_metadata)
        if overlap:
            raise ValueError(f"extra golden metadata overlaps reserved keys: {sorted(overlap)}")
        metadata.update(extra_metadata)
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return json_path, npz_path


def validate_stories_golden(index_path: str | Path) -> dict:
    """Validate the archive and every tensor referenced by a golden index."""

    index_path = Path(index_path)
    metadata = json.loads(index_path.read_text(encoding="utf-8"))
    supported_formats = {
        "budgetcuda-stories-fp32-golden",
        "budgetcuda-stories-wxay-golden",
    }
    if metadata.get("format") not in supported_formats or metadata.get("version") != 1:
        raise ValueError("unsupported TinyStories golden format or version")
    npz_name = metadata.get("npz_file")
    if not isinstance(npz_name, str) or Path(npz_name).name != npz_name:
        raise ValueError("golden npz_file must be a sibling filename")
    npz_path = index_path.parent / npz_name
    if not npz_path.is_file():
        raise ValueError(f"golden archive not found: {npz_path}")
    archive_hash = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    if archive_hash != metadata.get("npz_sha256"):
        raise ValueError("golden archive SHA-256 mismatch")

    expected: dict[str, dict] = {}
    for step in metadata.get("steps", []):
        for tensor in step.get("tensors", []):
            key = tensor.get("npz_key")
            if not isinstance(key, str) or key in expected:
                raise ValueError(f"invalid or duplicate golden tensor key: {key!r}")
            expected[key] = tensor
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            missing = sorted(set(expected) - set(archive.files))
            extra = sorted(set(archive.files) - set(expected))
            raise ValueError(f"golden tensor keys differ: missing={missing}, extra={extra}")
        for key, tensor in expected.items():
            value = np.ascontiguousarray(archive[key])
            if list(value.shape) != tensor.get("shape"):
                raise ValueError(f"golden tensor {key} shape mismatch")
            if str(value.dtype) != tensor.get("dtype"):
                raise ValueError(f"golden tensor {key} dtype mismatch")
            digest = hashlib.sha256(value.tobytes()).hexdigest()
            if digest != tensor.get("sha256"):
                raise ValueError(f"golden tensor {key} SHA-256 mismatch")
    return {
        "index": str(index_path),
        "archive": str(npz_path),
        "steps": len(metadata.get("steps", [])),
        "tensors": len(expected),
        "output_tokens": metadata.get("output_tokens", []),
        "decoded_text": metadata.get("decoded_text"),
        "npz_sha256": archive_hash,
    }
