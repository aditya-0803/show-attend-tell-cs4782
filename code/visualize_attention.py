"""Make the poster-friendly attention visualizations.

Given an image and a trained checkpoint, decodes a caption with beam search and saves
a grid of panels: one small panel per word, showing the input image overlaid with
that word's 14x14 attention map upsampled to image resolution.

Usage:
    python -m code.visualize_attention \\
        --checkpoint checkpoints/run1/best.pt \\
        --image path/to/image.jpg \\
        --vocab data/vocab.json \\
        --beam 3 \\
        --out results/attn_example.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.transform import resize as sk_resize

from .data.dataset import image_transform
from .data.vocab import Vocabulary
from .generate import beam_search
from .models.captioner import ShowAttendTell
from .models.encoder import VGGEncoder
from .utils.checkpoint import load_checkpoint


def _overlay(ax, img: np.ndarray, alpha_map: np.ndarray, word: str) -> None:
    ax.imshow(img)
    ax.imshow(alpha_map, alpha=0.7, cmap="jet")
    ax.set_title(word, fontsize=14)
    ax.axis("off")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--beam", type=int, default=3)
    p.add_argument("--out", type=Path, default=Path("results/attn_example.png"))
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary.load(args.vocab)

    # Load model + encoder.
    encoder = VGGEncoder().to(device).eval()
    model = ShowAttendTell(vocab_size=len(vocab), pad_id=vocab.pad_id).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    # Load image.
    with Image.open(args.image) as im:
        im_rgb = im.convert("RGB")
    tf = image_transform(train=False)
    img_t = tf(im_rgb).unsqueeze(0).to(device)

    # Forward + beam search.
    with torch.inference_mode():
        annotations = encoder(img_t)                     # (1, 196, 512)
        words, alphas = beam_search(model, annotations, vocab, beam_size=args.beam)

    # alphas: (T, 196) -> upsample each to image resolution for display.
    img_disp = np.asarray(im_rgb.resize((224, 224)))
    grids: List[np.ndarray] = []
    for a in alphas:
        a = a.view(14, 14).numpy()
        a = sk_resize(a, (224, 224), order=3, mode="reflect", anti_aliasing=True)
        grids.append(a)

    # Plot <start> panel + one per word.
    n = len(grids)
    cols = min(n + 1, 6)
    rows = int(np.ceil((n + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.atleast_2d(axes)

    axes[0, 0].imshow(img_disp)
    axes[0, 0].set_title("input")
    axes[0, 0].axis("off")

    for idx, (w, g) in enumerate(zip(words + (["<end>"] if len(words) < n else []), grids)):
        r, c = divmod(idx + 1, cols)
        _overlay(axes[r, c], img_disp, g, w)

    # Hide any remaining empty panels.
    for k in range(n + 1, rows * cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")

    plt.suptitle(" ".join(words), fontsize=16, y=1.02)
    plt.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"[viz] wrote {args.out}")
    print("caption:", " ".join(words))


if __name__ == "__main__":
    main()
