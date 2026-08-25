"""Validate hashes, W4/W8 packing, padding, and integer GEMM arithmetic."""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path


def decode_w4(raw: bytes, k: int, n: int) -> list[list[int]]:
    matrix = [[0] * n for _ in range(k)]
    cursor = 0
    for n0 in range(0, n, 8):
        for inner in range(k):
            for pair in range(4):
                byte = raw[cursor]; cursor += 1
                for half in range(2):
                    column = n0 + pair * 2 + half
                    nibble = (byte >> (4 * half)) & 15
                    value = nibble if nibble < 8 else nibble - 16
                    if column < n: matrix[inner][column] = value
                    elif value != 0: raise ValueError("Non-zero padded W4 lane")
    if cursor != len(raw): raise ValueError("Unexpected packed W4 byte count")
    return matrix


def decode_w8(raw: bytes, k: int, n: int) -> list[list[int]]:
    matrix = [[0] * n for _ in range(k)]
    cursor = 0
    for n0 in range(0, n, 8):
        for inner in range(k):
            for lane in range(8):
                value = raw[cursor]; cursor += 1
                value = value if value < 128 else value - 256
                column = n0 + lane
                if column < n: matrix[inner][column] = value
                elif value != 0: raise ValueError("Non-zero padded W8 lane")
    if cursor != len(raw): raise ValueError("Unexpected packed W8 byte count")
    return matrix


def validate(manifest_path: Path, samples: int = 8) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")); rng = random.Random(42)
    weight_bits = int(manifest.get("weight_bits", 4)); observed_bits = set(); max_accumulator_error = 0; packed_bytes = 0
    for name, tensor in manifest["tensors"].items():
        raw = (manifest_path.parent / tensor["file"]).read_bytes(); packed_bytes += len(raw)
        if hashlib.sha256(raw).hexdigest() != tensor["sha256"]: raise ValueError(f"Hash mismatch: {name}")
        k, n = tensor["shape_k_n"]
        tensor_bits = int(tensor.get("weight_bits", weight_bits)); observed_bits.add(tensor_bits)
        weights = decode_w4(raw, k, n) if tensor_bits == 4 else decode_w8(raw, k, n)
        for _ in range(samples):
            activation = [rng.randint(-128, 127) for _ in range(k)]
            for column in rng.sample(range(n), min(4, n)):
                scalar = sum(activation[i] * weights[i][column] for i in range(k))
                tiled = 0
                n0 = column // 8
                lane = column & 7
                for i in range(k):
                    if tensor_bits == 4:
                        byte = raw[(n0 * k + i) * 4 + lane // 2]
                        nibble = (byte >> (4 * (lane & 1))) & 15
                        value = nibble if nibble < 8 else nibble - 16
                    else:
                        byte = raw[(n0 * k + i) * 8 + lane]
                        value = byte if byte < 128 else byte - 256
                    tiled += activation[i] * value
                max_accumulator_error = max(max_accumulator_error, abs(scalar - tiled))
    result = {"weight_bits": sorted(observed_bits), "tensor_count": len(manifest["tensors"]), "packed_weight_bytes": packed_bytes,
              "integer_gemm_samples_per_tensor": samples, "max_accumulator_error": max_accumulator_error,
              "status": "pass" if max_accumulator_error == 0 else "fail"}
    return result


def main():
    here=Path(__file__).resolve().parent;parser=argparse.ArgumentParser();parser.add_argument("--manifest",type=Path,default=here/"artifacts"/"budgetcuda_w8a8"/"manifest.json");parser.add_argument("--output",type=Path);args=parser.parse_args();result=validate(args.manifest);output=args.output or args.manifest.with_name("conformance.json");output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2),encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
