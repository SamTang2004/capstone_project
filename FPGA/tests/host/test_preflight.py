from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from budgetcuda.compiler import build_tiny_int4_cnn_package
from budgetcuda.dtypes import DType
from budgetcuda.preflight import validate_model_package


def tiny_package():
    return build_tiny_int4_cnn_package(
        (1, 1, 3, 3),
        [1, -1, 2, 0],
        multiplier=1,
        right_shift=0,
        kernel_shape=(2, 2),
    )


def test_preflight_accepts_current_int4_cnn():
    report = validate_model_package(tiny_package())
    assert report.layers == 3
    assert report.buffers == 6
    assert report.arena_bytes_required <= report.arena_bytes_available


def test_preflight_rejects_arena_overflow():
    with pytest.raises(ValueError, match="firmware arena"):
        validate_model_package(tiny_package(), arena_bytes=64)


def test_preflight_rejects_int8_deployment_package():
    package = tiny_package()
    package.buffers[package.input_index] = package.buffers[package.input_index].__class__(
        name="bad-input",
        shape=package.buffers[package.input_index].shape,
        dtype=DType.INT8,
        layout=package.buffers[package.input_index].layout,
        usage=package.buffers[package.input_index].usage,
    )
    with pytest.raises(ValueError, match="packed INT4 model input"):
        validate_model_package(package)
