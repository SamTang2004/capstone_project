"""Validate board probe payloads/goldens independently with the host backend."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[3] / 'testing/tiny_stories/probe_overlay_contract.py'
spec = importlib.util.spec_from_file_location('overlay_probe', path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize('case,epsilon', [
    ('layernorm', 0), ('layernorm', 1), ('layernorm', 80),
    ('gemm-w8', 0), ('gemm-w4', 0),
])
def test_probe_golden_matches_mock(case, epsilon):
    package, payload, expected = probe.make_case(case, epsilon)
    device = probe.Device.mock()
    model = device.load_model(package)
    try:
        assert len(package.layers) == 1
        assert probe.raw(model.run(payload)) == expected
    finally:
        model.close()
        device.close()
