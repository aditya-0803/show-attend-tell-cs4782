"""Caption generation: greedy and beam search.

Both operate on a trained ``ShowAttendTell`` model given either an image tensor or
precomputed annotations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .data.vocab import Vocabulary
from .models.captioner import ShowAttendTell


@dataclass
class BeamHypothesis:
    tokens: List[int]
    alphas: List[torch.Tensor]
    logp: float


@torch.inference_mode()
def greedy_caption(
    model: ShowAttendTell,
    annotations: torch.Tensor,          # (1, L, D_a) or (L, D_a)
    vocab: Vocabulary,
    max_len: int = 20,
) -> Tuple[List[str], torch.Tensor]:
    """Greedy decoding. Returns (tokens, alphas of shape (T, L))."""
    if annotations.dim() == 2:
        annotations = annotations.unsqueeze(0)
    device = annotations.device

    h, c = model.decoder.init_hidden(annotations)
    prev = torch.tensor([vocab.start_id], device=device)
    tokens: list[int] = []
    alphas: list[torch.Tensor] = []
    for _ in range(max_len):
        logits, alpha, (h, c) = model.decoder.step(prev, annotations, (h, c))
        nxt = int(logits.argmax(dim=-1).item())
        alphas.append(alpha.squeeze(0).detach().cpu())
        if nxt == vocab.end_id:
            break
        tokens.append(nxt)
        prev = torch.tensor([nxt], device=device)
    words = vocab.decode(tokens, strip_special=True)
    return words, torch.stack(alphas) if alphas else torch.empty(0)


@torch.inference_mode()
def beam_search(
    model: ShowAttendTell,
    annotations: torch.Tensor,          # (1, L, D_a) or (L, D_a)
    vocab: Vocabulary,
    beam_size: int = 3,
    max_len: int = 20,
) -> Tuple[List[str], torch.Tensor]:
    """Beam search. Returns the best hypothesis as (tokens, alphas (T, L))."""
    if annotations.dim() == 2:
        annotations = annotations.unsqueeze(0)
    device = annotations.device
    L, D_a = annotations.size(1), annotations.size(2)

    # Expand annotations to beam_size.
    ann = annotations.expand(beam_size, L, D_a).contiguous()

    h, c = model.decoder.init_hidden(ann)
    prev = torch.full((beam_size,), vocab.start_id, dtype=torch.long, device=device)

    # Log-probs of each beam. Initially only beam 0 is "live"; others start at -inf
    # so they don't all produce the same sequence in step 1.
    seq_logp = torch.full((beam_size,), float("-inf"), device=device)
    seq_logp[0] = 0.0
    seqs: list[list[int]] = [[] for _ in range(beam_size)]
    alpha_seqs: list[list[torch.Tensor]] = [[] for _ in range(beam_size)]

    finished: list[BeamHypothesis] = []

    for t in range(max_len):
        logits, alpha, (h, c) = model.decoder.step(prev, ann, (h, c))
        logp = F.log_softmax(logits, dim=-1)                  # (beam, V)
        total = seq_logp.unsqueeze(-1) + logp                  # (beam, V)
        flat = total.view(-1)                                   # (beam*V,)
        top_vals, top_idx = flat.topk(beam_size)                # best beam*V -> beam
        beam_idx = torch.div(top_idx, model.vocab_size, rounding_mode="floor")
        tok_idx = top_idx % model.vocab_size

        new_seqs, new_alpha_seqs = [], []
        new_h, new_c = [], []
        new_prev = []
        new_logp = []
        for i in range(beam_size):
            bi = int(beam_idx[i].item())
            ti = int(tok_idx[i].item())
            if ti == vocab.end_id:
                finished.append(
                    BeamHypothesis(
                        tokens=seqs[bi] + [],
                        alphas=alpha_seqs[bi] + [alpha[bi].detach().cpu()],
                        logp=float(top_vals[i].item()),
                    )
                )
                # Keep this beam "alive" but with -inf so it won't be selected again.
                new_seqs.append(seqs[bi] + [ti])
                new_alpha_seqs.append(alpha_seqs[bi] + [alpha[bi].detach().cpu()])
                new_h.append(h[bi])
                new_c.append(c[bi])
                new_prev.append(ti)
                new_logp.append(float("-inf"))
            else:
                new_seqs.append(seqs[bi] + [ti])
                new_alpha_seqs.append(alpha_seqs[bi] + [alpha[bi].detach().cpu()])
                new_h.append(h[bi])
                new_c.append(c[bi])
                new_prev.append(ti)
                new_logp.append(float(top_vals[i].item()))

        seqs = new_seqs
        alpha_seqs = new_alpha_seqs
        h = torch.stack(new_h, dim=0)
        c = torch.stack(new_c, dim=0)
        prev = torch.tensor(new_prev, dtype=torch.long, device=device)
        seq_logp = torch.tensor(new_logp, device=device)

        if len(finished) >= beam_size and torch.isinf(seq_logp).all():
            break

    if not finished:
        # Nothing finished before max_len; use the best unfinished beam.
        best_i = int(seq_logp.argmax().item())
        finished.append(
            BeamHypothesis(
                tokens=seqs[best_i], alphas=alpha_seqs[best_i],
                logp=float(seq_logp[best_i].item()),
            )
        )

    # Length-normalize (standard trick to avoid bias toward short sequences).
    finished.sort(
        key=lambda h: h.logp / max(len(h.tokens), 1),
        reverse=True,
    )
    best = finished[0]
    words = vocab.decode(best.tokens, strip_special=True)
    a = torch.stack(best.alphas) if best.alphas else torch.empty(0)
    return words, a
