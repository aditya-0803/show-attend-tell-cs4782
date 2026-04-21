"""Run beam-search decoding over the Karpathy test split and report metrics.

Usage:
    python -m code.evaluate \\
        --checkpoint checkpoints/run1/best.pt \\
        --features features/coco_vgg16.h5 \\
        --vocab data/vocab.json \\
        --splits data/karpathy_splits/dataset_coco.json \\
        --beam 3 \\
        --out results/test_metrics.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from .data.dataset import CocoCaptionFeatureDataset, load_karpathy
from .data.vocab import Vocabulary
from .generate import beam_search
from .models.captioner import ShowAttendTell
from .utils.checkpoint import load_checkpoint
from .utils.metrics import compute_all_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--max-caption-len", type=int, default=22)
    p.add_argument("--out", type=Path, default=Path("results/test_metrics.json"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary.load(args.vocab)

    samples = load_karpathy(args.splits, vocab, args.split, max_length=args.max_caption_len)
    ds = CocoCaptionFeatureDataset(samples, args.features)

    model = ShowAttendTell(vocab_size=len(vocab), pad_id=vocab.pad_id).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    refs: dict[int, list[str]] = defaultdict(list)
    for s in samples:
        refs[s.image_id].append(s.caption)

    seen: set[int] = set()
    hyps: dict[int, str] = {}
    for i, s in enumerate(tqdm(samples, desc=f"decode/{args.split}")):
        if s.image_id in seen:
            continue
        seen.add(s.image_id)
        feats, _, _ = ds[i]
        feats = feats.to(device).unsqueeze(0)
        words, _ = beam_search(model, feats, vocab, beam_size=args.beam)
        hyps[s.image_id] = " ".join(words)

    scores = compute_all_metrics(refs, hyps, bleu_brevity_penalty=False)
    print(json.dumps(scores, indent=2))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"split": args.split, "beam": args.beam, "scores": scores, "hyps": hyps}, f, indent=2)
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
