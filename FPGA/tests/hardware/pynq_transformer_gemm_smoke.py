"""Board-level W8/W4 GEMM and requantize-pack smoke test.

Copy this script, the ``budgetcuda`` package, and the generated BIT/HWH pair to
the PYNQ-Z2. Run as root so CMA allocation and MMIO access are available.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda.dtypes import DType
from budgetcuda.pynq import (
    PynqPlRuntime,
    configure_pynq_z2_fclk0,
    pack_transformer_activations_int8,
    pack_transformer_weights,
)


def _bytes_view(buffer):
    storage = getattr(buffer.data, "data", buffer.data)
    return memoryview(storage).cast("B")


def _write_bytes(buffer, payload: bytes) -> None:
    view = _bytes_view(buffer)
    view[: len(payload)] = payload


def _read_i32(buffer, count: int) -> list[int]:
    return list(struct.unpack_from(f"<{count}i", _bytes_view(buffer), 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bitfile", default="bcuda_pynq_z2.bit")
    parser.add_argument("--legacy-xlnk", action="store_true", help="Use an already-programmed image and /dev/xlnk")
    parser.add_argument("--xrt-direct", action="store_true", help="Use an already-programmed image with ZOCL/XRT and direct MMIO")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--columns", type=int, choices=(2, 3), default=3)
    parser.add_argument("--mode", choices=("w8", "w4", "both"), default="both")
    args = parser.parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")

    configure_pynq_z2_fclk0(100)
    if args.legacy_xlnk and args.xrt_direct:
        raise SystemExit("Select only one direct allocator")
    if args.legacy_xlnk:
        runtime = PynqPlRuntime.legacy_xlnk()
    elif args.xrt_direct:
        runtime = PynqPlRuntime.xrt_direct()
    else:
        runtime = PynqPlRuntime(args.bitfile)
    a_values = [[1, 2], [3, 4], [5, 6]]
    # Keep the common test matrix representable by both signed INT8 and INT4
    # so both datapaths can be compared against exactly the same golden result.
    b_values = [[7, -3, -2], [4, 5, 3]]
    if args.columns == 2:
        b_values = [row[:2] for row in b_values]
        expected = [15, 7, 37, 11, 59, 15]
    else:
        expected = [15, 7, 4, 37, 11, 6, 59, 15, 8]
    packed_a_bytes = pack_transformer_activations_int8(a_values, 3, 2)
    expected_packed = pack_transformer_activations_int8(
        [expected[row * args.columns:(row + 1) * args.columns] for row in range(3)], 3, args.columns
    )

    packed_a = runtime.allocate((len(packed_a_bytes),), DType.INT8)
    packed_b8 = runtime.allocate((16,), DType.INT8)
    packed_b4 = runtime.allocate((8,), DType.INT8)
    accum = runtime.allocate((3 * args.columns,), DType.INT32)
    repacked = runtime.allocate((8 * args.columns,), DType.INT8)
    try:
        print(
            "buffers "
            f"A=0x{packed_a.physical_address:08x} "
            f"W8=0x{packed_b8.physical_address:08x} "
            f"W4=0x{packed_b4.physical_address:08x} "
            f"C=0x{accum.physical_address:08x} "
            f"packed=0x{repacked.physical_address:08x}"
        )
        _write_bytes(packed_a, packed_a_bytes)
        _write_bytes(packed_b8, pack_transformer_weights(b_values, 2, args.columns))
        _write_bytes(packed_b4, pack_transformer_weights(b_values, 2, args.columns, int4=True))
        # Debug-enable selects single-step execution. Leave it disabled for
        # free-running endurance while still collecting all cycle counters.
        runtime.set_debug(enabled=False, clear=True)

        modes = {
            "w8": ((packed_b8, False),),
            "w4": ((packed_b4, True),),
            "both": ((packed_b8, False), (packed_b4, True)),
        }[args.mode]
        for iteration in range(args.iterations):
            for weights, int4 in modes:
                runtime.submit_transformer_gemm(packed_a, weights, accum, 3, args.columns, 2, weight_int4=int4)
                try:
                    runtime.wait_gemm(accum)
                except Exception:
                    print(f"failed iteration={iteration} mode=W{4 if int4 else 8}")
                    print(runtime.debug_snapshot())
                    accum.invalidate()
                    print(f"partial_output={_read_i32(accum, 3 * args.columns)}")
                    raise
                actual = _read_i32(accum, 3 * args.columns)
                if actual != expected:
                    raise RuntimeError(f"iteration={iteration} W{4 if int4 else 8} mismatch: {actual}")

        runtime.submit_transformer_requantize_pack(accum, repacked, 3, args.columns, multiplier=1, right_shift=0)
        runtime.wait_gemm(repacked)
        actual_packed = bytes(_bytes_view(repacked)[:8 * args.columns])
        if actual_packed != expected_packed:
            raise RuntimeError(f"requantize-pack mismatch: {actual_packed.hex()}")

        counters = runtime.debug_snapshot()
        mode_summary = " + ".join(f"{args.iterations} W{4 if int4 else 8}" for _, int4 in modes)
        print(f"PASS: {mode_summary} GEMMs and fused requantize-pack")
        print(" ".join(f"{key}={value}" for key, value in counters.items()))
    finally:
        for buffer in (packed_a, packed_b8, packed_b4, accum, repacked):
            buffer.close()
        runtime.close()


if __name__ == "__main__":
    main()
