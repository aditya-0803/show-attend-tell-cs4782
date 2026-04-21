"""Show, Attend and Tell — full model tying the encoder, attention and decoder together.

Two modes:

* ``encode_then_decode(images, captions, lengths)``: end-to-end from pixels. Used when
  no precomputed features are available.
* ``decode(annotations, captions, lengths)``: decoder-only from precomputed annotations.
  This is the fast path used during training after ``precompute_features.py`` has run.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .decoder import AttentionDecoder, DecoderOutput
from .encoder import VGGEncoder


class ShowAttendTell(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        feature_dim: int = 512,
        attn_dim: int = 512,
        dropout: float = 0.5,
        pad_id: int = 0,
        include_encoder: bool = False,
    ):
        super().__init__()
        self.include_encoder = include_encoder
        self.encoder: Optional[VGGEncoder] = (
            VGGEncoder() if include_encoder else None
        )
        self.decoder = AttentionDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            attn_dim=attn_dim,
            dropout=dropout,
            pad_id=pad_id,
        )

    # Convenience property.
    @property
    def vocab_size(self) -> int:
        return self.decoder.vocab_size

    def decode(
        self,
        annotations: torch.Tensor,
        captions: torch.Tensor,
        lengths: torch.Tensor,
    ) -> DecoderOutput:
        return self.decoder(annotations, captions, lengths)

    def encode_then_decode(
        self,
        images: torch.Tensor,
        captions: torch.Tensor,
        lengths: torch.Tensor,
    ) -> DecoderOutput:
        if self.encoder is None:
            raise RuntimeError(
                "Model was built without an encoder. Pass `include_encoder=True` or "
                "provide precomputed annotations."
            )
        ann = self.encoder(images)
        return self.decoder(ann, captions, lengths)

    # Default forward dispatches to whichever path makes sense.
    def forward(self, inputs, captions, lengths):
        if inputs.dim() == 4:   # (B, 3, H, W)
            return self.encode_then_decode(inputs, captions, lengths)
        return self.decode(inputs, captions, lengths)
