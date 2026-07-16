from __future__ import annotations

import argparse
import json
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent / "golden_traces"


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def int8_pattern(index: int) -> int:
    value = ((index * 37 + 19) % 255) - 127
    return max(-128, min(127, value))


def matrix(rows: int, cols: int, offset: int) -> list[list[int]]:
    return [[int8_pattern(offset + r * cols + c) for c in range(cols)] for r in range(rows)]


def matmul_reference(A: list[list[int]], B: list[list[int]]) -> list[list[int]]:
    M = len(A)
    K = len(A[0]) if A else 0
    N = len(B[0]) if B else 0
    return [
        [sum(int(A[m][k]) * int(B[k][n]) for k in range(K)) for n in range(N)]
        for m in range(M)
    ]


def zero_matrix(rows: int, cols: int) -> list[list[int]]:
    return [[0 for _ in range(cols)] for _ in range(rows)]


def tile_extent(index: int, total: int, tile_size: int) -> int:
    start = index * tile_size
    return max(0, min(tile_size, total - start))


def build_schedule(M: int, N: int, K: int, D: int, array_count: int) -> list[dict]:
    m_tiles = ceil_div(M, D)
    n_tiles = ceil_div(N, D)
    k_tiles = ceil_div(K, D)
    schedule = []
    cycle = 0

    for mt in range(m_tiles):
        m0 = mt * D
        m_len = tile_extent(mt, M, D)

        for ng in range(0, n_tiles, array_count):
            active_n_tiles = list(range(ng, min(ng + array_count, n_tiles)))

            for kt in range(k_tiles):
                k0 = kt * D
                k_len = tile_extent(kt, K, D)
                arrays = []

                for array_id, nt in enumerate(active_n_tiles):
                    n0 = nt * D
                    n_len = tile_extent(nt, N, D)
                    arrays.append(
                        {
                            "array_id": array_id,
                            "n_tile": nt,
                            "n0": n0,
                            "n_len": n_len,
                            "B_tile": {
                                "row0": k0,
                                "col0": n0,
                                "rows": k_len,
                                "cols": n_len,
                                "pad_rows": D - k_len,
                                "pad_cols": D - n_len,
                            },
                            "C_tile": {
                                "row0": m0,
                                "col0": n0,
                                "rows": m_len,
                                "cols": n_len,
                            },
                        }
                    )

                schedule.append(
                    {
                        "step": cycle,
                        "m_tile": mt,
                        "k_tile": kt,
                        "n_tile_group_start": ng,
                        "clear_accumulators": kt == 0,
                        "write_c_tiles": kt == k_tiles - 1,
                        "A_tile": {
                            "row0": m0,
                            "col0": k0,
                            "rows": m_len,
                            "cols": k_len,
                            "pad_rows": D - m_len,
                            "pad_cols": D - k_len,
                        },
                        "arrays": arrays,
                    }
                )
                cycle += 1

    return schedule


def simulate_tiled_gemm(A: list[list[int]], B: list[list[int]], D: int, array_count: int) -> tuple[list[list[int]], list[dict]]:
    M = len(A)
    K = len(A[0]) if A else 0
    N = len(B[0]) if B else 0
    schedule = build_schedule(M, N, K, D, array_count)
    C = zero_matrix(M, N)

    for step in schedule:
        m0 = step["A_tile"]["row0"]
        k0 = step["A_tile"]["col0"]
        m_len = step["A_tile"]["rows"]
        k_len = step["A_tile"]["cols"]

        if step["clear_accumulators"]:
            for array in step["arrays"]:
                n0 = array["C_tile"]["col0"]
                n_len = array["C_tile"]["cols"]
                for m in range(m0, m0 + m_len):
                    for n in range(n0, n0 + n_len):
                        C[m][n] = 0

        for array in step["arrays"]:
            n0 = array["B_tile"]["col0"]
            n_len = array["B_tile"]["cols"]
            for mi in range(m_len):
                for ni in range(n_len):
                    C[m0 + mi][n0 + ni] += sum(
                        int(A[m0 + mi][k0 + kk]) * int(B[k0 + kk][n0 + ni])
                        for kk in range(k_len)
                    )

    return C, schedule


def flatten(matrix_: list[list[int]]) -> list[int]:
    return [value for row in matrix_ for value in row]


def write_hex(path: Path, values: list[int], width: int) -> None:
    mask = (1 << (width * 4)) - 1
    path.write_text("\n".join(f"{value & mask:0{width}x}" for value in values) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate tiled GEMM controller scheduling.")
    parser.add_argument("--M", type=int, default=35)
    parser.add_argument("--N", type=int, default=41)
    parser.add_argument("--K", type=int, default=27)
    parser.add_argument("--D", type=int, default=16)
    parser.add_argument("--arrays", type=int, default=2)
    parser.add_argument("--prefix", default="tiled_gemm_controller")
    args = parser.parse_args()

    if args.M <= 0 or args.N <= 0 or args.K <= 0 or args.D <= 0 or args.arrays <= 0:
        raise ValueError("M, N, K, D, and arrays must all be positive")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    A = matrix(args.M, args.K, 0)
    B = matrix(args.K, args.N, 4096)
    expected = matmul_reference(A, B)
    actual, schedule = simulate_tiled_gemm(A, B, args.D, args.arrays)

    if actual != expected:
        raise AssertionError("tiled GEMM controller simulation did not match reference GEMM")

    base = OUT_DIR / args.prefix
    json_path = base.with_suffix(".json")
    a_path = OUT_DIR / f"{args.prefix}_A.hex"
    b_path = OUT_DIR / f"{args.prefix}_B.hex"
    c_path = OUT_DIR / f"{args.prefix}_C.hex"

    json_path.write_text(
        json.dumps(
            {
                "M": args.M,
                "N": args.N,
                "K": args.K,
                "D": args.D,
                "array_count": args.arrays,
                "m_tiles": ceil_div(args.M, args.D),
                "n_tiles": ceil_div(args.N, args.D),
                "k_tiles": ceil_div(args.K, args.D),
                "schedule_steps": len(schedule),
                "schedule": schedule,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_hex(a_path, flatten(A), 2)
    write_hex(b_path, flatten(B), 2)
    write_hex(c_path, flatten(actual), 8)

    print(f"Wrote {json_path}")
    print(f"Wrote {a_path}")
    print(f"Wrote {b_path}")
    print(f"Wrote {c_path}")
    print(f"Verified tiled GEMM: M={args.M} N={args.N} K={args.K} D={args.D} arrays={args.arrays} steps={len(schedule)}")


if __name__ == "__main__":
    main()
