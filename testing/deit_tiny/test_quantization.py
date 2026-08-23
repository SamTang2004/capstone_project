from __future__ import annotations

import torch
import torch.nn as nn

from quantization import (
    FP4E2M1FakeQuantizer,
    FakeQuantizer,
    QuantizedLinear,
    prepare_qat_model,
)


def test_signed_int4_has_fifteen_symmetric_levels() -> None:
    quantizer = FakeQuantizer(4, signed=True)
    quantizer.train()
    values = torch.linspace(-1.0, 1.0, 1001)
    output = quantizer(values)
    assert output.unique().numel() == 15
    assert torch.isclose(output.min(), -output.max())


def test_unsigned_int4_has_sixteen_levels() -> None:
    quantizer = FakeQuantizer(4, signed=False)
    quantizer.train()
    values = torch.linspace(0.0, 1.0, 1001)
    output = quantizer(values)
    assert output.unique().numel() == 16
    assert output.min() == 0


def test_quantized_linear_preserves_shape_and_gradients() -> None:
    layer = QuantizedLinear(nn.Linear(8, 4), bits=4)
    layer.train()
    inputs = torch.randn(3, 8, requires_grad=True)
    output = layer(inputs)
    output.sum().backward()
    assert output.shape == (3, 4)
    assert inputs.grad is not None
    assert layer.weight.grad is not None


def test_w4a8_uses_separate_weight_and_activation_widths() -> None:
    model = prepare_qat_model(nn.Sequential(nn.Linear(8, 4)), 4, 8)
    layer = model[0]
    assert isinstance(layer, QuantizedLinear)
    assert layer.weight_quant.bits == 4
    assert layer.input_quant.bits == 8
    assert layer.output_quant.bits == 8


def test_fp4_e2m1_uses_standard_codebook_and_ste() -> None:
    quantizer = FP4E2M1FakeQuantizer(signed=True)
    quantizer.train()
    values = torch.tensor(
        [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
         0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        requires_grad=True,
    )
    output = quantizer(values)
    assert torch.equal(output, values)
    output.sum().backward()
    assert torch.equal(values.grad, torch.ones_like(values))
