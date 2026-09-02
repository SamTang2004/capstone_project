import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda.stories import (  # noqa: E402
    Llama2CCheckpoint,
    Llama2CTokenizer,
    StoriesFloatReference,
    export_stories_golden,
    validate_stories_golden,
)
from budgetcuda.stories_quantization import (  # noqa: E402
    StoriesQuantizationSpec,
    evaluate_quantization_sweep,
    export_quantized_stories_golden,
    write_quantization_report,
)


def _checkpoint_arrays(config, *, shared=True):
    dim, hidden, layers, heads, kv_heads, vocab, seq_len = config
    head_size = dim // heads
    kv_dim = head_size * kv_heads
    embedding = np.arange(vocab * dim, dtype=np.float32).reshape(vocab, dim) / 10.0
    arrays = {
        "token_embedding_table": embedding,
        "rms_att_weight": np.ones((layers, dim), dtype=np.float32),
        "wq": np.zeros((layers, dim, dim), dtype=np.float32),
        "wk": np.zeros((layers, kv_dim, dim), dtype=np.float32),
        "wv": np.zeros((layers, kv_dim, dim), dtype=np.float32),
        "wo": np.zeros((layers, dim, dim), dtype=np.float32),
        "rms_ffn_weight": np.ones((layers, dim), dtype=np.float32),
        "w1": np.zeros((layers, hidden, dim), dtype=np.float32),
        "w2": np.zeros((layers, dim, hidden), dtype=np.float32),
        "w3": np.zeros((layers, hidden, dim), dtype=np.float32),
        "rms_final_weight": np.ones(dim, dtype=np.float32),
        "legacy_rope_real": np.zeros((seq_len, head_size // 2), dtype=np.float32),
        "legacy_rope_imag": np.zeros((seq_len, head_size // 2), dtype=np.float32),
    }
    if not shared:
        arrays["wcls"] = np.flip(embedding, axis=0).copy()
    return arrays


def _write_checkpoint(path, config=(4, 6, 1, 2, 1, 8, 5), *, shared=True):
    arrays = _checkpoint_arrays(config, shared=shared)
    header = list(config)
    if not shared:
        header[5] = -header[5]
    with path.open("wb") as output:
        output.write(struct.pack("<7i", *header))
        for array in arrays.values():
            output.write(np.asarray(array, dtype="<f4").tobytes())
    return arrays


def _write_random_checkpoint(path, config=(4, 6, 1, 2, 1, 8, 8)):
    arrays = _checkpoint_arrays(config, shared=True)
    rng = np.random.default_rng(7)
    for name in ("wq", "wk", "wv", "wo", "w1", "w2", "w3"):
        arrays[name][...] = rng.normal(0.0, 0.15, arrays[name].shape).astype(np.float32)
    with path.open("wb") as output:
        output.write(struct.pack("<7i", *config))
        for array in arrays.values():
            output.write(np.asarray(array, dtype="<f4").tobytes())


def _write_tokenizer(path):
    pieces = ["<unk>", "<s>", "</s>", " ", "a", "b", "ab"]
    scores = [-10.0, -10.0, -10.0, 0.0, 1.0, 1.0, 10.0]
    encoded = [piece.encode("utf-8") for piece in pieces]
    with path.open("wb") as output:
        output.write(struct.pack("<I", max(map(len, encoded))))
        for score, raw in zip(scores, encoded):
            output.write(struct.pack("<fI", score, len(raw)))
            output.write(raw)
    return pieces


def test_checkpoint_reader_maps_legacy_layout_and_reports_ledger(tmp_path):
    path = tmp_path / "tiny.bin"
    arrays = _write_checkpoint(path)
    checkpoint = Llama2CCheckpoint(path)

    assert checkpoint.config.dim == 4
    assert checkpoint.config.kv_dim == 2
    assert checkpoint.config.shared_classifier
    assert np.shares_memory(checkpoint.wcls, checkpoint.token_embedding_table)
    np.testing.assert_array_equal(checkpoint.w1, arrays["w1"])
    assert checkpoint.tensor_records[-1].name == "legacy_rope_imag"
    ledger = checkpoint.ledger()
    assert ledger["file_size_bytes"] == path.stat().st_size
    assert ledger["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert ledger["tensors"][0]["offset_bytes"] == 28


def test_checkpoint_reader_maps_unshared_classifier_and_rejects_bad_size(tmp_path):
    path = tmp_path / "unshared.bin"
    arrays = _write_checkpoint(path, shared=False)
    checkpoint = Llama2CCheckpoint(path)
    assert not checkpoint.config.shared_classifier
    np.testing.assert_array_equal(checkpoint.wcls, arrays["wcls"])
    assert checkpoint.tensor_records[-1].name == "wcls"

    truncated = tmp_path / "truncated.bin"
    truncated.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(ValueError, match="truncated"):
        Llama2CCheckpoint(truncated)


def test_stories_shape_validation_is_explicit(tmp_path):
    path = tmp_path / "fixture.bin"
    _write_checkpoint(path)
    with pytest.raises(ValueError, match="not the stories15M model"):
        Llama2CCheckpoint(path, require_stories15m=True)


def test_tokenizer_parses_merges_and_decodes_llama2c_format(tmp_path):
    path = tmp_path / "tokenizer.bin"
    pieces = _write_tokenizer(path)
    tokenizer = Llama2CTokenizer(path, len(pieces))
    tokens = tokenizer.encode("ab", bos=True, eos=False)
    assert tokens == [1, 3, 6]
    assert tokenizer.decode(tokens) == "ab"


def test_float_reference_produces_expected_logits_and_sequential_cache(tmp_path):
    path = tmp_path / "tiny.bin"
    arrays = _write_checkpoint(path)
    checkpoint = Llama2CCheckpoint(path)
    reference = StoriesFloatReference(checkpoint)

    result = reference.forward(3)
    embedding = arrays["token_embedding_table"][3]
    expected_norm = embedding / np.sqrt(np.mean(embedding * embedding) + 1e-5)
    expected_logits = arrays["token_embedding_table"] @ expected_norm
    np.testing.assert_allclose(result.logits, expected_logits, rtol=1e-6, atol=1e-6)
    assert result.trace["layer_00.attention_probabilities"].shape == (2, 1)
    assert result.trace["layer_00.attention_logits"].shape == (2, 1)
    assert result.trace["layer_00.post_attention"].shape == (4,)
    assert result.trace["layer_00.key_cache"].shape == (1, 2)
    assert result.trace["layer_00.value_cache"].shape == (1, 2)
    assert result.trace["layer_00.output"].shape == (4,)
    assert reference.next_position == 1
    with pytest.raises(ValueError, match="sequential position"):
        reference.forward(4, position=0)


def test_golden_export_contains_full_logits_hashes_and_token_flow(tmp_path):
    checkpoint_path = tmp_path / "tiny.bin"
    _write_checkpoint(checkpoint_path)
    checkpoint = Llama2CCheckpoint(checkpoint_path)
    json_path, npz_path = export_stories_golden(
        checkpoint, tmp_path / "goldens" / "sample", [1, 3], top_k=3,
    )

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["format"] == "budgetcuda-stories-fp32-golden"
    assert metadata["prompt_tokens"] == [1, 3]
    assert len(metadata["steps"]) == 2
    assert metadata["steps"][0]["actual_next_token"] == 3
    assert metadata["steps"][1]["actual_next_token"] is None
    assert len(metadata["steps"][0]["top_logits"]) == 3
    assert metadata["npz_sha256"] == hashlib.sha256(npz_path.read_bytes()).hexdigest()
    with np.load(npz_path) as golden:
        assert golden["step_000__logits"].shape == (8,)
        assert golden["step_001__layer_00__swiglu"].shape == (6,)
        assert golden["step_001__layer_00__key_cache"].shape == (2, 2)
    validation = validate_stories_golden(json_path)
    assert validation["steps"] == 2
    assert validation["tensors"] > 0


def test_golden_validator_rejects_modified_index(tmp_path):
    checkpoint_path = tmp_path / "tiny.bin"
    _write_checkpoint(checkpoint_path)
    json_path, _ = export_stories_golden(
        Llama2CCheckpoint(checkpoint_path), tmp_path / "sample", [1], top_k=2,
    )
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    metadata["npz_sha256"] = "0" * 64
    json_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="archive SHA-256 mismatch"):
        validate_stories_golden(json_path)


def test_wxay_sweep_grades_low_precision_against_fp32(tmp_path):
    checkpoint_path = tmp_path / "random.bin"
    _write_random_checkpoint(checkpoint_path)
    checkpoint = Llama2CCheckpoint(checkpoint_path)
    specs = [StoriesQuantizationSpec(8, 8), StoriesQuantizationSpec(2, 2)]
    report = evaluate_quantization_sweep(
        checkpoint, [[1, 2, 3, 4, 5]], specs,
        sample_prompt_tokens=[1], sample_new_tokens=2,
    )
    assert report["prediction_tokens"] == 4
    assert report["candidate_float_perplexity"] == report["baseline_perplexity"]
    assert report["baseline_checkpoint_sha256"] == report["checkpoint_sha256"]
    assert [item["label"] for item in report["evaluations"]] == ["W8A8", "W2A2"]
    high, low = report["evaluations"]
    assert high["mean_normalized_rmse"] < low["mean_normalized_rmse"]
    assert high["mean_logit_cosine"] > low["mean_logit_cosine"]
    assert low["grade"] in ("A", "B", "C", "F")
    json_path, markdown_path = write_quantization_report(report, tmp_path / "sweep")
    assert json.loads(json_path.read_text(encoding="utf-8"))["prediction_tokens"] == 4
    assert "| W8A8 |" in markdown_path.read_text(encoding="utf-8")


def test_wxay_golden_records_policy_and_validates_payload(tmp_path):
    checkpoint_path = tmp_path / "random.bin"
    _write_random_checkpoint(checkpoint_path)
    checkpoint = Llama2CCheckpoint(checkpoint_path)
    spec = StoriesQuantizationSpec(4, 8, "per_output")
    json_path, npz_path = export_quantized_stories_golden(
        checkpoint, tmp_path / "w4a8", [1, 2], spec,
        max_new_tokens=1, provenance={"kind": "unit-test QAT"},
    )
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["format"] == "budgetcuda-stories-wxay-golden"
    assert metadata["quantization"]["spec"] == spec.as_dict()
    assert metadata["provenance"]["kind"] == "unit-test QAT"
    assert metadata["npz_sha256"] == hashlib.sha256(npz_path.read_bytes()).hexdigest()
    validation = validate_stories_golden(json_path)
    assert validation["steps"] == 2
    assert validation["tensors"] > 0
