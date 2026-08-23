import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda import DeiTManifestRunner, Device
from budgetcuda.packing import pack_transformer_weights


def _tensor(root, manifest, name, values, k, n, bias=None):
    raw = pack_transformer_weights(values, k, n)
    filename = name.replace(".", "_") + ".w8.bin"
    (root / filename).write_bytes(raw)
    manifest["tensors"][name] = {
        "file": filename,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shape_k_n": [k, n],
        "input_scale": 1.0,
        "output_scale": 1.0,
        "requant": [1, 0, 0, 0],
        "bias_int32": [] if bias is None else bias,
    }


def test_manifest_runner_executes_every_encoder_stage(tmp_path):
    hidden, mlp = 6, 8
    manifest = {
        "format": "budgetcuda-deit-w8a8", "version": 1, "weight_bits": 8,
        "classes": ["fresh", "rotten"], "tokens": 2, "hidden_dim": hidden,
        "heads": 3, "depth": 1, "mlp_dim": mlp, "tensors": {},
        "blocks": [{
            "index": 0, "norm1_gamma": [1.0] * hidden, "norm1_beta": [0.0] * hidden,
            "norm2_gamma": [1.0] * hidden, "norm2_beta": [0.0] * hidden,
        }],
        "final_norm": {"gamma": [1.0] * hidden, "beta": [0.0] * hidden},
    }
    zero_h = [[0] * hidden for _ in range(hidden)]
    for label in ("q", "k", "v", "proj"):
        _tensor(tmp_path, manifest, f"blocks.0.attn.{label}", zero_h, hidden, hidden, [0] * hidden)
    _tensor(tmp_path, manifest, "blocks.0.mlp.fc1", [[0] * mlp for _ in range(hidden)], hidden, mlp, [0] * mlp)
    _tensor(tmp_path, manifest, "blocks.0.mlp.fc2", [[0] * hidden for _ in range(mlp)], mlp, hidden, [0] * hidden)
    _tensor(tmp_path, manifest, "head", [[0, 0] for _ in range(hidden)], hidden, 2, [1, 4])
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = DeiTManifestRunner(Device.mock(), manifest_path).run_tokens([-3, -2, -1, 1, 2, 3] * 2)
    assert result.class_index == 1
    assert result.class_name == "rotten"
    assert result.logits == (1, 4)
    assert [name for name, _ in result.stage_seconds] == [
        "block_0.attention", "block_0.mlp", "final_layernorm", "six_class_head",
    ]
