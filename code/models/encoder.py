"""VGG-16 encoder that produces 14x14x512 feature maps from 224x224 inputs.

Matches the proposal: "a pretrained VGG-16 (torchvision) and extract the 14 x 14 x 512
feature maps from the final convolutional layer before max pooling, yielding 196
annotation vectors of dimension 512."

We slice ``vgg16.features`` at index 29. That index is inclusive of ``conv5_3`` and its
ReLU, and exclusive of the final maxpool at index 30. For a 224x224 input this yields
(B, 512, 14, 14).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16


class VGGEncoder(nn.Module):
    """Frozen VGG-16 convolutional feature extractor."""

    OUT_HW = 14
    OUT_C = 512

    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = vgg16(weights=weights)
        # features[:29] -> through conv5_3 + ReLU, before the final maxpool.
        self.features = nn.Sequential(*list(backbone.features.children())[:29])

        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.features.eval()

    def train(self, mode: bool = True):
        # Keep frozen conv in eval mode regardless of outer train() calls so that any
        # BN-like layers behave consistently. VGG16 has no BN so this is mostly a
        # no-op, but it's good hygiene for dropout-containing variants.
        super().train(mode)
        for p in self.parameters():
            if not p.requires_grad:
                self.features.eval()
                break
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, 224, 224) -> (B, 196, 512)."""
        x = self.features(images)                         # (B, 512, 14, 14)
        B, C, H, W = x.shape
        assert C == self.OUT_C and H == W == self.OUT_HW, (
            f"Unexpected feature shape {x.shape}; input must be 224x224."
        )
        return x.flatten(2).transpose(1, 2).contiguous()  # (B, 196, 512)
