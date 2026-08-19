"""Generate a deterministic INT8 Conv2D host-preslice trace for RTL tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from budgetcuda.compiler import preslice_conv2d_weights_int8


def main() -> None:
    out_channels, input_channels, kernel_h, kernel_w = 18, 2, 3, 3
    reduction = input_channels * kernel_h * kernel_w
    raw = bytes((oc * 29 + k * 3) & 0xFF for oc in range(out_channels) for k in range(reduction))
    packed = preslice_conv2d_weights_int8(
        raw,
        out_channels=out_channels,
        input_channels=input_channels,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
    )
    trace = {
        "layout": "n_tile,k_tile,k_lane,n_lane",
        "logical_shape_oihw": [out_channels, input_channels, kernel_h, kernel_w],
        "physical_shape": [2, 2, 16, 16],
        "logical_bytes": len(raw),
        "physical_bytes": len(packed),
        "tile_bytes": 256,
        "checks": {
            "tile_0_k0_oc0": packed[0],
            "tile_0_k0_oc1": packed[1],
            "tile_0_k17_oc0": packed[256 + 16],
            "tile_1_k0_oc16": packed[2 * 256],
            "padded_oc31_k17": packed[3 * 256 + 16 + 15],
        },
    }
    output = Path(__file__).with_name("golden_traces") / "conv_weight_preslice_trace.json"
    output.write_text(json.dumps(trace, indent=2) + "\n", encoding="ascii")
    print(f"wrote {output}: {len(raw)} logical bytes -> {len(packed)} tiled bytes")


if __name__ == "__main__":
    main()
