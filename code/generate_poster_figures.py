"""Produce every poster-ready figure from a trained checkpoint.

Run after ``code.train`` and ``code.evaluate`` have both completed. Reads:

* the checkpoint             (for attention viz)
* ``train_log.jsonl``        (for training-curve plot)
* ``test_metrics.json``      (for the results table and per-image captions)
* ``dataset_coco.json``      (for ground-truth references)
* ``features/coco_vgg16.h5`` (for attention viz — fast path)
* ``coco2014/``              (for raw images to overlay)

Outputs everything under ``--out-dir``:

    training_curves.png
    results_table.png
    results_table.md
    sample_captions_good.png
    sample_captions_bad.png
    caption_length_hist.png
    attention_examples/{00..19}.png

Usage:
    python -m code.generate_poster_figures \\
        --checkpoint /path/to/best.pt \\
        --features   /path/to/coco_vgg16.h5 \\
        --vocab      /path/to/vocab.json \\
        --splits     /path/to/dataset_coco.json \\
        --coco-root  /path/to/coco2014 \\
        --metrics-json /path/to/test_metrics.json \\
        --train-log  /path/to/train_log.jsonl \\
        --out-dir    /path/to/poster_figures \\
        --num-attn   20
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize as sk_resize

from .data.dataset import image_transform, load_karpathy
from .data.vocab import Vocabulary
from .generate import beam_search
from .models.captioner import ShowAttendTell
from .models.encoder import VGGEncoder
from .utils.checkpoint import load_checkpoint


# Paper reference numbers (Xu et al. 2015, Table 1, MS COCO, soft attention).
PAPER_NUMBERS = {
    "BLEU-1": 70.7,
    "BLEU-2": 49.2,
    "BLEU-3": 34.4,
    "BLEU-4": 24.3,
    "METEOR": 23.90,
}


# ----------------------------------------------------------------------------- #
# 1. Training curves                                                            #
# ----------------------------------------------------------------------------- #

def plot_training_curves(train_log_path: Path, out_path: Path) -> None:
    """Plot train xent (per epoch) and val BLEU-4 / METEOR from train_log.jsonl."""
    if not train_log_path.exists():
        print(f"[warn] {train_log_path} not found; skipping training-curve plot.")
        return
    records = []
    with open(train_log_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print(f"[warn] {train_log_path} is empty; skipping.")
        return

    epochs = [r["epoch"] for r in records]
    train_xent = [r["train"]["xent"] for r in records]
    val = [r["val"] for r in records]
    bleu4 = [v.get("BLEU-4") for v in val]
    meteor = [v.get("METEOR") for v in val]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_xent, "o-", color="#1f77b4", linewidth=2)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train cross-entropy")
    axes[0].set_title("Training loss")
    axes[0].grid(alpha=0.3)

    ax = axes[1]
    if any(b is not None for b in bleu4):
        ax.plot(epochs, bleu4, "o-", color="#ff7f0e", linewidth=2, label="val BLEU-4")
    if any(m is not None for m in meteor):
        ax.plot(epochs, meteor, "s-", color="#2ca02c", linewidth=2, label="val METEOR")
    ax.set_xlabel("epoch"); ax.set_ylabel("score")
    ax.set_title("Validation metrics")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


# ----------------------------------------------------------------------------- #
# 2. Results table                                                              #
# ----------------------------------------------------------------------------- #

def write_results_table(metrics_json: Path, out_md: Path, out_png: Path) -> None:
    """Write a paper-vs-ours comparison as both markdown and a PNG table."""
    with open(metrics_json) as f:
        data = json.load(f)
    ours = data.get("scores", {})
    rows = []
    for k in ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "METEOR"):
        paper = PAPER_NUMBERS.get(k, None)
        our = ours.get(k, None)
        rows.append((k, paper, our))

    # Markdown.
    lines = ["| Metric | Paper (soft) | Ours |", "|---|---:|---:|"]
    for k, p, o in rows:
        p_s = f"{p:.2f}" if p is not None else "—"
        o_s = f"{o:.2f}" if o is not None else "—"
        lines.append(f"| {k} | {p_s} | {o_s} |")
    out_md.write_text("\n".join(lines) + "\n")
    print(f"[viz] wrote {out_md}")

    # PNG (using matplotlib's table renderer — easy to drop onto a poster).
    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    ax.axis("off")
    cell_text = [
        [r[0], f"{r[1]:.2f}" if r[1] is not None else "—",
         f"{r[2]:.2f}" if r[2] is not None else "—"]
        for r in rows
    ]
    table = ax.table(
        cellText=cell_text,
        colLabels=["Metric", "Paper (soft)", "Ours"],
        cellLoc="center", loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 1.6)
    # Header row styling.
    for c in range(3):
        table[0, c].set_facecolor("#4a6fa5")
        table[0, c].set_text_props(color="white", weight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out_png}")


# ----------------------------------------------------------------------------- #
# 3. Sample caption panels (good + bad)                                         #
# ----------------------------------------------------------------------------- #

def _per_caption_bleu4(ref_tokens: List[List[str]], hyp_tokens: List[str]) -> float:
    """Sentence-level BLEU-4 with light smoothing, no brevity penalty."""
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    if not hyp_tokens:
        return 0.0
    try:
        return float(sentence_bleu(
            ref_tokens, hyp_tokens,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=SmoothingFunction().method1,
        ))
    except Exception:
        return 0.0


def _find_image_path(coco_root: Path, image_id: int, splits_json: Path) -> Path | None:
    """Look up the on-disk filename for a COCO image id."""
    with open(splits_json) as f:
        data = json.load(f)
    for img in data["images"]:
        if img["cocoid"] == image_id:
            return coco_root / img["filepath"] / img["filename"]
    return None


def plot_sample_captions(
    metrics_json: Path,
    splits_json: Path,
    coco_root: Path,
    out_good: Path,
    out_bad: Path,
    n_per_set: int = 6,
) -> None:
    """Pick n best- and n worst-BLEU-4 captions and make a grid figure for each."""
    with open(metrics_json) as f:
        data = json.load(f)
    hyps: Dict[str, str] = data.get("hyps", {})
    if not hyps:
        print("[warn] metrics_json has no hyps; skipping sample-caption figures.")
        return

    # Load references for every image in hyps.
    with open(splits_json) as f:
        ds = json.load(f)
    refs: Dict[int, List[str]] = defaultdict(list)
    filepath: Dict[int, str] = {}
    for img in ds["images"]:
        filepath[img["cocoid"]] = f"{img['filepath']}/{img['filename']}"
        for s in img["sentences"]:
            refs[img["cocoid"]].append(s["raw"])

    # Score every hypothesis.
    scored: list[tuple[int, str, float]] = []
    for k, h in hyps.items():
        img_id = int(k)
        if img_id not in refs:
            continue
        ref_tok = [r.lower().split() for r in refs[img_id]]
        hyp_tok = h.lower().split()
        s = _per_caption_bleu4(ref_tok, hyp_tok)
        scored.append((img_id, h, s))

    scored.sort(key=lambda t: t[2], reverse=True)
    # Take the top n "good" and bottom n "bad" — but skip trivial hyps like empty strings.
    good = [t for t in scored if len(t[1].split()) >= 3][:n_per_set]
    bad = [
        t for t in reversed(scored) if len(t[1].split()) >= 3
    ][:n_per_set]

    def _grid(items, title, out_path):
        cols = 3
        rows = int(np.ceil(len(items) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        axes = np.atleast_2d(axes)
        for idx, (img_id, hyp, score) in enumerate(items):
            r, c = divmod(idx, cols)
            p = coco_root / filepath[img_id]
            try:
                with Image.open(p) as im:
                    axes[r, c].imshow(im.convert("RGB"))
            except Exception:
                axes[r, c].text(0.5, 0.5, "(img missing)", ha="center")
            axes[r, c].axis("off")
            # Show the top ref + our hyp.
            ref0 = refs[img_id][0] if refs[img_id] else ""
            axes[r, c].set_title(
                f"ours: {hyp}\nref:  {ref0}\nBLEU-4={score:.2f}",
                fontsize=10, loc="left",
            )
        for k in range(len(items), rows * cols):
            r, c = divmod(k, cols)
            axes[r, c].axis("off")
        plt.suptitle(title, fontsize=16, y=1.01)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] wrote {out_path}")

    _grid(good, "Correct / near-correct captions", out_good)
    _grid(bad, "Failure cases", out_bad)


# ----------------------------------------------------------------------------- #
# 4. Caption length histogram                                                   #
# ----------------------------------------------------------------------------- #

def plot_length_histogram(
    metrics_json: Path, splits_json: Path, out_path: Path
) -> None:
    with open(metrics_json) as f:
        data = json.load(f)
    hyps: Dict[str, str] = data.get("hyps", {})
    if not hyps:
        return
    hyp_lens = [len(h.split()) for h in hyps.values()]

    with open(splits_json) as f:
        ds = json.load(f)
    ref_lens = []
    hyp_ids = {int(k) for k in hyps}
    for img in ds["images"]:
        if img["cocoid"] not in hyp_ids:
            continue
        for s in img["sentences"]:
            ref_lens.append(len(s["raw"].split()))

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.arange(0, 25)
    ax.hist(ref_lens, bins=bins, alpha=0.55, color="#6baed6", label="reference")
    ax.hist(hyp_lens, bins=bins, alpha=0.7, color="#fd8d3c", label="ours")
    ax.set_xlabel("caption length (words)")
    ax.set_ylabel("count")
    ax.set_title("Caption length distribution (test split)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out_path}")


# ----------------------------------------------------------------------------- #
# 5. Attention visualizations                                                   #
# ----------------------------------------------------------------------------- #

def _overlay_attention(ax, img_arr, alpha_map, title):
    ax.imshow(img_arr)
    ax.imshow(alpha_map, alpha=0.65, cmap="jet")
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def render_attention_visualizations(
    checkpoint: Path,
    features_h5: Path,
    splits_json: Path,
    vocab: Vocabulary,
    coco_root: Path,
    out_dir: Path,
    n_images: int = 20,
    beam_size: int = 3,
    seed: int = 42,
) -> None:
    import h5py

    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick diverse test images.
    with open(splits_json) as f:
        ds = json.load(f)
    test_imgs = [img for img in ds["images"] if img["split"] == "test"]
    random.seed(seed)
    picks = random.sample(test_imgs, min(n_images, len(test_imgs)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ShowAttendTell(vocab_size=len(vocab), pad_id=vocab.pad_id).to(device)
    load_checkpoint(checkpoint, model, map_location=device)
    model.eval()

    with h5py.File(str(features_h5), "r") as h5:
        for i, img_meta in enumerate(picks):
            img_id = img_meta["cocoid"]
            key = str(img_id)
            if key not in h5:
                print(f"[warn] features missing for {img_id}; skipping")
                continue
            feats = np.asarray(h5[key], dtype=np.float32)
            if feats.ndim == 3:
                feats = feats.reshape(-1, feats.shape[-1])
            annotations = torch.from_numpy(feats).unsqueeze(0).to(device)

            with torch.inference_mode():
                words, alphas = beam_search(
                    model, annotations, vocab, beam_size=beam_size,
                )

            # Image for display.
            img_path = coco_root / img_meta["filepath"] / img_meta["filename"]
            try:
                with Image.open(img_path) as im:
                    img_disp = np.asarray(im.convert("RGB").resize((224, 224)))
            except Exception as e:
                print(f"[warn] could not open {img_path}: {e}")
                continue

            # Upsample alphas to image resolution.
            grids = []
            for a in alphas:
                a = a.view(14, 14).numpy()
                a = sk_resize(a, (224, 224), order=3, mode="reflect", anti_aliasing=True)
                grids.append(a)

            # Layout.
            panels = len(grids)
            cols = min(panels + 1, 6)
            rows = int(np.ceil((panels + 1) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
            axes = np.atleast_2d(axes)
            axes[0, 0].imshow(img_disp); axes[0, 0].set_title("input"); axes[0, 0].axis("off")
            for idx, (w, g) in enumerate(zip(words, grids)):
                r, c = divmod(idx + 1, cols)
                _overlay_attention(axes[r, c], img_disp, g, w)
            for k in range(panels + 1, rows * cols):
                r, c = divmod(k, cols)
                axes[r, c].axis("off")
            plt.suptitle(" ".join(words), fontsize=14, y=1.03)
            plt.tight_layout()
            plt.savefig(out_dir / f"{i:02d}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

    print(f"[viz] wrote {n_images} attention figures to {out_dir}")


# ----------------------------------------------------------------------------- #
# Main                                                                          #
# ----------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",   type=Path, required=True)
    p.add_argument("--features",     type=Path, required=True)
    p.add_argument("--vocab",        type=Path, required=True)
    p.add_argument("--splits",       type=Path, required=True)
    p.add_argument("--coco-root",    type=Path, required=True)
    p.add_argument("--metrics-json", type=Path, required=True)
    p.add_argument("--train-log",    type=Path, required=True)
    p.add_argument("--out-dir",      type=Path, required=True)
    p.add_argument("--num-attn",     type=int, default=20)
    p.add_argument("--beam-size",    type=int, default=3)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    vocab = Vocabulary.load(args.vocab)

    plot_training_curves(args.train_log, args.out_dir / "training_curves.png")
    write_results_table(
        args.metrics_json,
        args.out_dir / "results_table.md",
        args.out_dir / "results_table.png",
    )
    plot_sample_captions(
        args.metrics_json, args.splits, args.coco_root,
        args.out_dir / "sample_captions_good.png",
        args.out_dir / "sample_captions_bad.png",
    )
    plot_length_histogram(
        args.metrics_json, args.splits, args.out_dir / "caption_length_hist.png",
    )
    render_attention_visualizations(
        args.checkpoint, args.features, args.splits, vocab, args.coco_root,
        args.out_dir / "attention_examples",
        n_images=args.num_attn, beam_size=args.beam_size,
    )
    print(f"\n[done] all figures in {args.out_dir}")


if __name__ == "__main__":
    main()
