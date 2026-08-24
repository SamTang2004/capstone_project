"""Hardware-oriented fake quantization for DeiT matrix operations."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class FakeQuantizer(nn.Module):
    """EMA-observed uniform fake quantizer with a straight-through estimator."""

    def __init__(
        self,
        bits: int,
        *,
        signed: bool = True,
        per_channel: bool = False,
        channel_axis: int = 0,
        momentum: float = 0.95,
    ) -> None:
        super().__init__()
        if bits < 2 or bits > 16:
            raise ValueError("bits must be in [2, 16]")
        self.bits = bits
        self.signed = signed
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        self.momentum = momentum
        self.observer_enabled = True
        self.fake_quant_enabled = True
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("initialized", torch.tensor(False))

    @property
    def quant_min(self) -> int:
        return -(2 ** (self.bits - 1) - 1) if self.signed else 0

    @property
    def quant_max(self) -> int:
        return 2 ** (self.bits - 1) - 1 if self.signed else 2**self.bits - 1

    def _observed_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        detached = tensor.detach().float()
        if self.per_channel:
            reduce_dims = tuple(dim for dim in range(detached.ndim) if dim != self.channel_axis)
            maximum = detached.abs().amax(dim=reduce_dims, keepdim=True)
        elif self.signed:
            maximum = detached.abs().amax().reshape(1)
        else:
            maximum = detached.amax().clamp_min(0).reshape(1)
        return (maximum / max(self.quant_max, 1)).clamp_min(1e-8)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.fake_quant_enabled:
            return tensor
        if self.observer_enabled and self.training:
            observed = self._observed_scale(tensor)
            if not bool(self.initialized):
                self.scale = observed.to(device=self.scale.device)
                self.initialized.fill_(True)
            else:
                self.scale.mul_(self.momentum).add_(
                    observed.to(device=self.scale.device), alpha=1.0 - self.momentum
                )
        scale = self.scale.to(device=tensor.device, dtype=tensor.dtype)
        quantized = torch.round(tensor / scale).clamp(self.quant_min, self.quant_max)
        dequantized = quantized * scale
        return tensor + (dequantized - tensor).detach()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        scale_key = prefix + "scale"
        if scale_key in state_dict and self.scale.shape != state_dict[scale_key].shape:
            self.scale = torch.empty_like(state_dict[scale_key])
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class FP4E2M1FakeQuantizer(nn.Module):
    """Scaled FP4 E2M1 fake quantizer using nearest-codebook rounding."""

    def __init__(
        self,
        *,
        signed: bool = True,
        per_channel: bool = False,
        channel_axis: int = 0,
        momentum: float = 0.95,
    ) -> None:
        super().__init__()
        self.signed = signed
        self.per_channel = per_channel
        self.channel_axis = channel_axis
        self.momentum = momentum
        self.observer_enabled = True
        self.fake_quant_enabled = True
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("initialized", torch.tensor(False))
        self.register_buffer(
            "magnitudes",
            torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]),
        )
        self.register_buffer(
            "thresholds",
            torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]),
        )

    def _observed_scale(self, tensor: torch.Tensor) -> torch.Tensor:
        detached = tensor.detach().float()
        if self.per_channel:
            reduce_dims = tuple(dim for dim in range(detached.ndim) if dim != self.channel_axis)
            maximum = detached.abs().amax(dim=reduce_dims, keepdim=True)
        elif self.signed:
            maximum = detached.abs().amax().reshape(1)
        else:
            maximum = detached.amax().clamp_min(0).reshape(1)
        return (maximum / 6.0).clamp_min(1e-8)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.fake_quant_enabled:
            return tensor
        if self.observer_enabled and self.training:
            observed = self._observed_scale(tensor)
            if not bool(self.initialized):
                self.scale = observed.to(device=self.scale.device)
                self.initialized.fill_(True)
            else:
                self.scale.mul_(self.momentum).add_(
                    observed.to(device=self.scale.device), alpha=1.0 - self.momentum
                )
        scale = self.scale.to(device=tensor.device, dtype=tensor.dtype)
        normalized = tensor / scale
        magnitudes = self.magnitudes.to(device=tensor.device, dtype=tensor.dtype)
        thresholds = self.thresholds.to(device=tensor.device, dtype=tensor.dtype)
        indices = torch.bucketize(normalized.abs().contiguous(), thresholds)
        quantized = magnitudes[indices]
        if self.signed:
            quantized = quantized.copysign(normalized)
        dequantized = quantized * scale
        return tensor + (dequantized - tensor).detach()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        scale_key = prefix + "scale"
        if scale_key in state_dict and self.scale.shape != state_dict[scale_key].shape:
            self.scale = torch.empty_like(state_dict[scale_key])
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class QuantizedLinear(nn.Module):
    def __init__(
        self, source: nn.Linear, bits: int, activation_bits: int | None = None,
        *, weight_per_channel: bool = True,
    ) -> None:
        super().__init__()
        activation_bits = bits if activation_bits is None else activation_bits
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.bias = (
            nn.Parameter(source.bias.detach().clone()) if source.bias is not None else None
        )
        self.input_quant = FakeQuantizer(activation_bits, signed=True)
        self.weight_quant = FakeQuantizer(
            bits, signed=True, per_channel=weight_per_channel, channel_axis=0
        )
        self.output_quant = FakeQuantizer(activation_bits, signed=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = F.linear(
            self.input_quant(tensor), self.weight_quant(self.weight), self.bias
        )
        return self.output_quant(output)


class QuantizedConv2d(nn.Module):
    def __init__(
        self, source: nn.Conv2d, bits: int, activation_bits: int | None = None,
        *, weight_per_channel: bool = True,
    ) -> None:
        super().__init__()
        activation_bits = bits if activation_bits is None else activation_bits
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.bias = (
            nn.Parameter(source.bias.detach().clone()) if source.bias is not None else None
        )
        self.input_quant = FakeQuantizer(activation_bits, signed=True)
        self.weight_quant = FakeQuantizer(
            bits, signed=True, per_channel=weight_per_channel, channel_axis=0
        )
        self.output_quant = FakeQuantizer(activation_bits, signed=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = F.conv2d(
            self.input_quant(tensor),
            self.weight_quant(self.weight),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return self.output_quant(output)


class FP4QuantizedLinear(nn.Module):
    def __init__(self, source: nn.Linear) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.bias = nn.Parameter(source.bias.detach().clone()) if source.bias is not None else None
        self.input_quant = FP4E2M1FakeQuantizer(signed=True)
        self.weight_quant = FP4E2M1FakeQuantizer(
            signed=True, per_channel=True, channel_axis=0
        )
        self.output_quant = FP4E2M1FakeQuantizer(signed=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = F.linear(
            self.input_quant(tensor), self.weight_quant(self.weight), self.bias
        )
        return self.output_quant(output)


class FP4QuantizedConv2d(nn.Module):
    def __init__(self, source: nn.Conv2d) -> None:
        super().__init__()
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.bias = nn.Parameter(source.bias.detach().clone()) if source.bias is not None else None
        self.input_quant = FP4E2M1FakeQuantizer(signed=True)
        self.weight_quant = FP4E2M1FakeQuantizer(
            signed=True, per_channel=True, channel_axis=0
        )
        self.output_quant = FP4E2M1FakeQuantizer(signed=True)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        output = F.conv2d(
            self.input_quant(tensor),
            self.weight_quant(self.weight),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return self.output_quant(output)


def _replace_matrix_modules(
    module: nn.Module, weight_bits: int, activation_bits: int,
    weight_per_channel: bool,
) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, QuantizedLinear(
                child, weight_bits, activation_bits,
                weight_per_channel=weight_per_channel))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, QuantizedConv2d(
                child, weight_bits, activation_bits,
                weight_per_channel=weight_per_channel))
        else:
            _replace_matrix_modules(
                child, weight_bits, activation_bits, weight_per_channel)


def _replace_fp4_matrix_modules(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, FP4QuantizedLinear(child))
        elif isinstance(child, nn.Conv2d):
            setattr(module, name, FP4QuantizedConv2d(child))
        else:
            _replace_fp4_matrix_modules(child)


def prepare_qat_model(
    float_model: nn.Module, bits: int, activation_bits: int | None = None,
    *, weight_per_channel: bool = True,
) -> nn.Module:
    """Clone a DeiT model and fake-quantize all matrix-operation operands.

    LayerNorm, GELU, softmax, residual additions, and accumulators remain in
    floating point. Q/K/V, attention probabilities, linear/conv weights, and
    linear/conv inputs and outputs are quantized to the requested precision.
    Weights use symmetric per-output-channel scales by default, or a hardware-
    compatible per-tensor scale when ``weight_per_channel`` is false.
    Activations use per-tensor scales. Attention probabilities use unsigned
    quantization.
    """
    activation_bits = bits if activation_bits is None else activation_bits
    model = copy.deepcopy(float_model)
    _replace_matrix_modules(model, bits, activation_bits, weight_per_channel)
    for block in getattr(model, "blocks", []):
        attention = block.attn
        attention.fused_attn = False
        attention.attn_drop = nn.Sequential(
            attention.attn_drop,
            FakeQuantizer(activation_bits, signed=False),
        )
    return model


def prepare_fp4_qat_model(float_model: nn.Module) -> nn.Module:
    """Clone a DeiT model and apply scaled E2M1 FP4 fake quantization."""
    model = copy.deepcopy(float_model)
    _replace_fp4_matrix_modules(model)
    for block in getattr(model, "blocks", []):
        attention = block.attn
        attention.fused_attn = False
        attention.attn_drop = nn.Sequential(
            attention.attn_drop,
            FP4E2M1FakeQuantizer(signed=False),
        )
    return model


def set_observers(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, (FakeQuantizer, FP4E2M1FakeQuantizer)):
            module.observer_enabled = enabled


def quantizer_count(model: nn.Module) -> int:
    return sum(
        isinstance(module, (FakeQuantizer, FP4E2M1FakeQuantizer))
        for module in model.modules()
    )
