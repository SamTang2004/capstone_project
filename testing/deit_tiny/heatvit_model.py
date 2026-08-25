"""HeatViT-style fixed-cardinality selectors for timm DeiT models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SelectorTrace:
    after_block: int
    input_tokens: int
    output_tokens: int
    selected_patch_indices: torch.Tensor


class LinearTokenSelector(nn.Module):
    """Hardware-oriented scorer with straight-through hard top-k selection.

    The inference value is exactly CLS + top-k patches + mean package token.
    A soft sigmoid gate and soft package are attached only to the backward pass,
    allowing classification/distillation losses to train the scorer despite the
    discrete top-k operation.
    """

    def __init__(self, hidden_dim: int, output_tokens: int, weight_bits: int = 8) -> None:
        super().__init__()
        if output_tokens < 2:
            raise ValueError("A selector output requires CLS and package tokens")
        self.output_tokens = int(output_tokens)
        self.weight_bits = int(weight_bits)
        if self.weight_bits not in (4, 8):
            raise ValueError("Selector scorer weights must be W4 or W8")
        self.score = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.score.weight, std=0.02)
        nn.init.zeros_(self.score.bias)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = tokens[:, 1:]
        keep = self.output_tokens - 2
        if keep <= 0 or keep >= patches.shape[1]:
            raise ValueError("Selector keep count must be inside the patch-token range")
        limit = (1 << (self.weight_bits - 1)) - 1
        scale = self.score.weight.detach().abs().max().clamp_min(1e-12) / limit
        quantized = torch.round(self.score.weight / scale).clamp(-limit - 1, limit) * scale
        quantized = self.score.weight + (quantized - self.score.weight).detach()
        scores = F.linear(patches, quantized, self.score.bias).squeeze(-1)
        probabilities = torch.sigmoid(scores)
        indices = scores.topk(keep, dim=1, sorted=False).indices
        indices = indices.sort(dim=1).values
        gather = indices.unsqueeze(-1).expand(-1, -1, patches.shape[-1])
        selected = patches.gather(1, gather)

        selected_probabilities = probabilities.gather(1, indices).unsqueeze(-1)
        selected = selected * (1.0 + selected_probabilities - selected_probabilities.detach())

        hard_removed = torch.ones_like(probabilities)
        hard_removed.scatter_(1, indices, 0.0)
        removed_count = hard_removed.sum(dim=1, keepdim=True).clamp_min(1.0)
        hard_package = (patches * hard_removed.unsqueeze(-1)).sum(dim=1, keepdim=True)
        hard_package = hard_package / removed_count.unsqueeze(-1)
        soft_removed = (1.0 - probabilities).unsqueeze(-1)
        soft_package = (patches * soft_removed).sum(dim=1, keepdim=True)
        soft_package = soft_package / soft_removed.sum(dim=1, keepdim=True).clamp_min(1e-6)
        package = hard_package + soft_package - soft_package.detach()
        return torch.cat((tokens[:, :1], selected, package), dim=1), indices


class HeatViTDeiT(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        selector_blocks: tuple[int, ...] = (2, 5, 8),
        output_tokens: tuple[int, ...] = (151, 106, 44),
        selector_bits: int = 8,
    ) -> None:
        super().__init__()
        if len(selector_blocks) != len(output_tokens) or tuple(sorted(selector_blocks)) != selector_blocks:
            raise ValueError("Selector blocks and token counts must be ordered and equally sized")
        hidden_dim = int(backbone.embed_dim)
        self.backbone = backbone
        self.selector_blocks = tuple(int(value) for value in selector_blocks)
        self.selectors = nn.ModuleList(
            LinearTokenSelector(hidden_dim, count, selector_bits) for count in output_tokens)
        self._selector_by_block = {block: index for index, block in enumerate(self.selector_blocks)}

    def selector_parameters(self):
        return self.selectors.parameters()

    def forward(self, inputs: torch.Tensor, *, return_trace: bool = False):
        x = self.backbone.patch_embed(inputs)
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        traces = []
        for block_index, block in enumerate(self.backbone.blocks):
            x = block(x)
            selector_index = self._selector_by_block.get(block_index)
            if selector_index is not None:
                input_tokens = x.shape[1]
                x, indices = self.selectors[selector_index](x)
                traces.append(SelectorTrace(block_index, input_tokens, x.shape[1], indices))
        x = self.backbone.norm(x)
        logits = self.backbone.forward_head(x, pre_logits=False)
        return (logits, tuple(traces)) if return_trace else logits
