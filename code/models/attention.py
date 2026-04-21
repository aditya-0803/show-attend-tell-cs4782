"""Soft attention module.

At every decoding step ``t`` we compute, for each of the L=196 annotation vectors a_i,
an unnormalized score:

    e_{t,i} = v^T tanh( W_a a_i + W_h h_{t-1} )                            (Eq. 4 in paper)

then

    alpha_{t,i} = softmax_i(e_{t,i})                                      (Eq. 5)

and the context vector is

    z_t = sum_i alpha_{t,i} * a_i                                          (Eq. 6, soft)

This module does the score+softmax, returning both alpha and z so the caller (the
decoder) can plug them into the LSTM step.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAttention(nn.Module):
    """Bahdanau-style additive attention over a fixed grid of annotation vectors."""

    def __init__(self, feature_dim: int = 512, hidden_dim: int = 512,
                 attn_dim: int = 512):
        super().__init__()
        self.W_a = nn.Linear(feature_dim, attn_dim, bias=False)
        self.W_h = nn.Linear(hidden_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, annotations: torch.Tensor, hidden: torch.Tensor):
        """
        Args:
            annotations: (B, L, D_a)  — spatial feature vectors from the encoder
            hidden:      (B, D_h)     — previous LSTM hidden state

        Returns:
            context: (B, D_a)         — weighted sum of annotations
            alpha:   (B, L)           — attention weights (sum to 1 along L)
        """
        # (B, L, attn) + (B, 1, attn) -> (B, L, attn)
        proj = torch.tanh(self.W_a(annotations) + self.W_h(hidden).unsqueeze(1))
        scores = self.v(proj).squeeze(-1)                  # (B, L)
        alpha = F.softmax(scores, dim=1)                   # (B, L)
        context = (alpha.unsqueeze(-1) * annotations).sum(dim=1)  # (B, D_a)
        return context, alpha
