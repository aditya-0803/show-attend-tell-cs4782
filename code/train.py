"""Training entry point for Show, Attend and Tell (soft attention).

Usage (precomputed features, recommended):
    python -m code.train \\
        --features features/coco_vgg16.h5 \\
        --splits data/karpathy_splits/dataset_coco.json \\
        --vocab data/vocab.json \\
        --out-dir checkpoints/run1

Usage (end-to-end from images, slow):
    python -m code.train --from-images \\
        --coco-root data/coco2014 \\
        --splits data/karpathy_splits/dataset_coco.json \\
        --vocab data/vocab.json \\
        --out-dir checkpoints/run1
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import (
    CocoCaptionFeatureDataset,
    CocoCaptionImageDataset,
    LengthBucketBatchSampler,
    collate_pad,
    load_karpathy,
)
from .data.vocab import Vocabulary
from .generate import beam_search
from .models.captioner import ShowAttendTell
from .utils.checkpoint import save_checkpoint
from .utils.metrics import compute_all_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, help="HDF5 file of precomputed VGG features")
    p.add_argument("--from-images", action="store_true", help="Train end-to-end from JPEGs")
    p.add_argument("--coco-root", type=Path, default=Path("data/coco2014"))
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--lr-decay", type=float, default=0.8)
    p.add_argument("--lr-decay-every", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--doubly-stochastic-lambda", type=float, default=1.0)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--max-caption-len", type=int, default=22)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--beam-size", type=int, default=3)
    p.add_argument("--patience", type=int, default=5,
                   help="Early stopping: #epochs without val BLEU-4 improvement.")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _build_loaders(args, vocab: Vocabulary):
    train_samples = load_karpathy(args.splits, vocab, "train", max_length=args.max_caption_len)
    val_samples = load_karpathy(args.splits, vocab, "val", max_length=args.max_caption_len)

    if args.from_images:
        train_ds = CocoCaptionImageDataset(train_samples, args.coco_root)
        val_ds = CocoCaptionImageDataset(val_samples, args.coco_root)
    else:
        assert args.features is not None, "--features is required unless --from-images"
        train_ds = CocoCaptionFeatureDataset(train_samples, args.features)
        val_ds = CocoCaptionFeatureDataset(val_samples, args.features)

    train_sampler = LengthBucketBatchSampler(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        seed=args.seed,
    )
    val_sampler = LengthBucketBatchSampler(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False,
    )

    train_loader = DataLoader(
        train_ds, batch_sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda b: collate_pad(b, pad_id=vocab.pad_id),
    )
    val_loader = DataLoader(
        val_ds, batch_sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=lambda b: collate_pad(b, pad_id=vocab.pad_id),
    )
    return train_loader, val_loader, train_sampler, val_ds


def _step_loss(
    model: ShowAttendTell,
    feats: torch.Tensor,
    caps: torch.Tensor,
    lens: torch.Tensor,
    criterion: nn.CrossEntropyLoss,
    ds_lambda: float,
):
    out = model(feats, caps, lens)
    logits = out.logits                               # (B, T, V)
    alphas = out.alphas                               # (B, T, L)
    dec_lens = out.lengths                            # (B,)

    # Targets are the ground-truth captions shifted by one (captions[:, 1:]).
    targets = caps[:, 1 : 1 + logits.size(1)]          # (B, T)
    B, T, V = logits.shape
    # Mask out padding in the loss using dec_lens.
    mask = torch.arange(T, device=logits.device).unsqueeze(0) < dec_lens.unsqueeze(1)
    loss_xent = criterion(
        logits.reshape(-1, V),
        targets.reshape(-1),
    )
    # The criterion already ignores pad_id; mask is for the doubly-stochastic term.

    # Doubly stochastic regularization: we want each spatial location to receive roughly
    # equal cumulative attention, i.e. sum_t alpha_{t,i} ~ 1 after correct normalization.
    # Following Eq. 14 of the paper: lambda * sum_i (1 - sum_t alpha_{t,i})^2
    masked_alphas = alphas * mask.unsqueeze(-1).float()
    sum_over_t = masked_alphas.sum(dim=1)              # (B, L)
    loss_ds = ((1.0 - sum_over_t) ** 2).sum(dim=1).mean()

    loss = loss_xent + ds_lambda * loss_ds
    return loss, loss_xent.detach(), loss_ds.detach()


def _run_validation(model, val_ds, vocab, device, beam_size: int, limit: int | None = None):
    """Run beam-search decoding on the val set and compute metrics."""
    model.eval()

    # Group ground-truth captions by image id.
    refs: dict[int, list[str]] = defaultdict(list)
    for s in val_ds.samples:
        refs[s.image_id].append(s.caption)

    # Deduplicate images (val contains the same image multiple times, once per caption).
    img_to_features: dict[int, torch.Tensor] = {}
    for i, s in enumerate(val_ds.samples):
        if s.image_id in img_to_features:
            continue
        feats, _, _ = val_ds[i]
        img_to_features[s.image_id] = feats

    if limit is not None:
        ids = list(img_to_features)[:limit]
    else:
        ids = list(img_to_features)

    hyps: dict[int, str] = {}
    for img_id in tqdm(ids, desc="val/beam"):
        feats = img_to_features[img_id].to(device).unsqueeze(0)   # (1, L, D_a)
        words, _ = beam_search(model, feats, vocab, beam_size=beam_size)
        hyps[img_id] = " ".join(words)

    refs = {k: refs[k] for k in ids}
    scores = compute_all_metrics(refs, hyps, bleu_brevity_penalty=False)
    model.train()
    return scores, hyps


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    vocab = Vocabulary.load(args.vocab)
    print(f"[vocab] size = {len(vocab)}")

    train_loader, val_loader, train_sampler, val_ds = _build_loaders(args, vocab)
    print(f"[data] train batches/epoch = {len(train_sampler)}")

    model = ShowAttendTell(
        vocab_size=len(vocab),
        dropout=args.dropout,
        pad_id=vocab.pad_id,
        include_encoder=args.from_images,
    ).to(device)

    # Only optimize trainable params (excludes frozen VGG).
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_every, gamma=args.lr_decay,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    best_bleu4 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        if args.from_images and model.encoder is not None:
            model.encoder.eval()   # keep frozen conv in eval mode

        t0 = time.time()
        running = {"loss": 0.0, "xent": 0.0, "ds": 0.0, "n": 0}
        for step, (feats, caps, lens) in enumerate(train_loader, start=1):
            feats = feats.to(device, non_blocking=True)
            caps = caps.to(device, non_blocking=True)
            lens = lens.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss, l_x, l_d = _step_loss(
                model, feats, caps, lens, criterion,
                ds_lambda=args.doubly_stochastic_lambda,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()

            running["loss"] += loss.item()
            running["xent"] += l_x.item()
            running["ds"] += l_d.item()
            running["n"] += 1
            if step % args.log_every == 0:
                avg = {k: running[k] / max(running["n"], 1) for k in ("loss", "xent", "ds")}
                print(f"  epoch {epoch} step {step}/{len(train_sampler)}  "
                      f"loss={avg['loss']:.3f}  xent={avg['xent']:.3f}  ds={avg['ds']:.3f}")

        scheduler.step()
        dt = time.time() - t0
        avg = {k: running[k] / max(running["n"], 1) for k in ("loss", "xent", "ds")}
        print(f"[epoch {epoch}] train loss={avg['loss']:.3f} "
              f"(xent={avg['xent']:.3f}, ds={avg['ds']:.3f})  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  elapsed={dt:.1f}s")

        # Validation with beam search on a limit for per-epoch speed, full eval only
        # at the very end. Limit keeps per-epoch val cost reasonable on Colab.
        val_limit = 500 if epoch < args.epochs else None
        scores, _ = _run_validation(
            model, val_ds, vocab, device,
            beam_size=args.beam_size, limit=val_limit,
        )
        score_str = "  ".join(f"{k}={v:.2f}" for k, v in scores.items())
        print(f"[epoch {epoch}] val ({val_limit or 'full'})  {score_str}")

        bleu4 = scores.get("BLEU-4", 0.0)
        is_best = bleu4 > best_bleu4
        if is_best:
            best_bleu4 = bleu4
            epochs_without_improvement = 0
            save_checkpoint(
                args.out_dir / "best.pt", model, optimizer, scheduler,
                epoch=epoch, best_metric=best_bleu4,
                extra={"val_scores": scores},
            )
            print(f"[epoch {epoch}] NEW BEST BLEU-4 = {best_bleu4:.2f}  -> saved.")
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            args.out_dir / "last.pt", model, optimizer, scheduler,
            epoch=epoch, best_metric=best_bleu4,
            extra={"val_scores": scores},
        )

        # Periodic JSON log for plotting.
        with open(args.out_dir / "train_log.jsonl", "a") as f:
            f.write(json.dumps({"epoch": epoch, "train": avg, "val": scores}) + "\n")

        if epochs_without_improvement >= args.patience:
            print(f"[stop] no BLEU-4 improvement for {args.patience} epochs; stopping.")
            break

    print(f"[done] best val BLEU-4 = {best_bleu4:.2f}")


if __name__ == "__main__":
    main()
