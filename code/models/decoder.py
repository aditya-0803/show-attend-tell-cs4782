"""Attention-based LSTM decoder.

Architecture (following Xu et al. 2015, soft-attention variant):

    * Embedding: V -> E_y  (E_y = embed_dim)
    * Initial state: h_0, c_0 = tanh(W_h_init * mean(a)),  tanh(W_c_init * mean(a))
    * At step t:
        context_t, alpha_t = Attention(a, h_{t-1})
        beta_t = sigmoid(W_beta * h_{t-1})         (gating scalar per sample)
        z_t = beta_t * context_t                   (gated context; Section 4.2.1)
        h_t, c_t = LSTMCell(input=[E y_{t-1}; z_t], (h_{t-1}, c_{t-1}))
        logits = W_out * ( E y_{t-1}  +  W_h_out h_t  +  W_z_out z_t )   (deep output; Eq. 7)

We use teacher forcing during training (feed ground-truth previous token) and emit
per-step alphas so the training loss can add a doubly-stochastic regularizer
`lambda * sum_i (1 - sum_t alpha_{t,i})^2` (Eq. 14).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .attention import SoftAttention


@dataclass
class DecoderOutput:
    logits: torch.Tensor       # (B, T, V)
    alphas: torch.Tensor       # (B, T, L)
    lengths: torch.Tensor      # (B,) — decoded lengths (excluding <start>, including <end> slot)


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 512,
        hidden_dim: int = 512,
        feature_dim: int = 512,
        attn_dim: int = 512,
        dropout: float = 0.5,
        pad_id: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.pad_id = pad_id

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)

        # Initial state projections from mean annotation (Section 4.2.1).
        self.init_h = nn.Linear(feature_dim, hidden_dim)
        self.init_c = nn.Linear(feature_dim, hidden_dim)

        self.attention = SoftAttention(feature_dim, hidden_dim, attn_dim)
        # Gating scalar f_beta(h_{t-1}).
        self.f_beta = nn.Linear(hidden_dim, feature_dim)

        self.lstm_cell = nn.LSTMCell(embed_dim + feature_dim, hidden_dim)

        # Deep output layer (Eq. 7): L_o ( E y + L_h h + L_z z )
        self.out_y = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_h = nn.Linear(hidden_dim, embed_dim, bias=False)
        self.out_z = nn.Linear(feature_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------ #
    # State helpers                                                      #
    # ------------------------------------------------------------------ #

    def init_hidden(self, annotations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize (h_0, c_0) from the mean annotation vector.

        annotations: (B, L, D_a) -> h0, c0 each (B, D_h)
        """
        mean_a = annotations.mean(dim=1)
        h = torch.tanh(self.init_h(mean_a))
        c = torch.tanh(self.init_c(mean_a))
        return h, c

    # ------------------------------------------------------------------ #
    # Single step (used by generation / beam search)                     #
    # ------------------------------------------------------------------ #

    def step(
        self,
        prev_token: torch.Tensor,               # (B,) long
        annotations: torch.Tensor,              # (B, L, D_a)
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run one decoding step. Returns (logits, alpha, new_state)."""
        h, c = state
        emb = self.embedding(prev_token)                        # (B, E)
        context, alpha = self.attention(annotations, h)         # (B, D_a), (B, L)
        gate = torch.sigmoid(self.f_beta(h))                    # (B, D_a)
        gated_ctx = gate * context                              # (B, D_a)

        lstm_in = torch.cat([emb, gated_ctx], dim=1)            # (B, E + D_a)
        h, c = self.lstm_cell(lstm_in, (h, c))

        pre_out = self.out_y(emb) + self.out_h(h) + self.out_z(gated_ctx)
        pre_out = self.dropout(torch.tanh(pre_out))
        logits = self.out_proj(pre_out)                         # (B, V)
        return logits, alpha, (h, c)

    # ------------------------------------------------------------------ #
    # Teacher-forced forward (training)                                  #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        annotations: torch.Tensor,   # (B, L, D_a)
        captions: torch.Tensor,      # (B, T) long — includes <start> at t=0, <end> before pad
        lengths: torch.Tensor,       # (B,) long — includes <start> and <end>
    ) -> DecoderOutput:
        """Teacher-forced decoding.

        The decoder consumes captions[:, :-1] (everything except the final token) and
        predicts captions[:, 1:]. So the number of decoding steps is max(lengths) - 1.
        """
        B, T = captions.shape
        device = captions.device
        # Steps = max target length excluding the last target-only position.
        decode_lengths = (lengths - 1).clamp(min=1)
        T_dec = int(decode_lengths.max().item())

        h, c = self.init_hidden(annotations)
        logits_out = torch.zeros(B, T_dec, self.vocab_size, device=device)
        alphas_out = torch.zeros(B, T_dec, annotations.size(1), device=device)

        # Captions are assumed sorted by decreasing length (our collate does this).
        for t in range(T_dec):
            active = int((decode_lengths > t).sum().item())
            if active == 0:
                break
            prev_tok = captions[:active, t]                 # (active,)
            h_a, c_a = h[:active], c[:active]
            ann_a = annotations[:active]
            logits, alpha, (h_a, c_a) = self.step(prev_tok, ann_a, (h_a, c_a))
            logits_out[:active, t] = logits
            alphas_out[:active, t] = alpha
            # Keep states aligned; inactive rows retain their old (unused) states.
            h = torch.cat([h_a, h[active:]], dim=0)
            c = torch.cat([c_a, c[active:]], dim=0)

        return DecoderOutput(logits=logits_out, alphas=alphas_out, lengths=decode_lengths)
