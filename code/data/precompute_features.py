"""Precompute VGG-16 conv5_3 features (14x14x512) for every COCO image and cache to HDF5.

This is the decisive speedup for multi-epoch training: the expensive conv forward pass
runs once, and the LSTM decoder can iterate on cached features for the rest of training.

Usage:
    python -m code.data.precompute_features \\
        --coco-root data/coco2014 \\
        --splits data/karpathy_splits/dataset_coco.json \\
        --out features/coco_vgg16.h5 \\
        --batch-size 64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..models.encoder import VGGEncoder
from .dataset import image_transform


class _ImageOnly(Dataset):
    """Reads raw images for precomputation, keyed by (image_id, filepath)."""

    def __init__(self, items, coco_root):
        self.items = items
        self.coco_root = Path(coco_root)
        self.tf = image_transform(train=False)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_id, fpath = self.items[idx]
        with Image.open(self.coco_root / fpath) as img:
            t = self.tf(img.convert("RGB"))
        return t, img_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16",
                        help="Storage dtype; fp16 halves disk usage with negligible loss.")
    args = parser.parse_args()

    with open(args.splits) as f:
        karpathy = json.load(f)

    # Deduplicate by image_id (same image appears once per split entry, not per caption).
    seen: set[int] = set()
    items = []
    for img in karpathy["images"]:
        if img["cocoid"] in seen:
            continue
        seen.add(img["cocoid"])
        items.append((img["cocoid"], f"{img['filepath']}/{img['filename']}"))
    print(f"[precompute] {len(items)} unique images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = VGGEncoder().to(device).eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # We write into an HDF5 with one dataset per image.
    np_dtype = np.float16 if args.dtype == "fp16" else np.float32

    loader = DataLoader(
        _ImageOnly(items, args.coco_root),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    with h5py.File(args.out, "w") as h5:
        with torch.inference_mode():
            for imgs, ids in tqdm(loader, desc="encode"):
                imgs = imgs.to(device, non_blocking=True)
                feats = encoder(imgs)              # (B, 196, 512)
                feats_np = feats.detach().cpu().numpy().astype(np_dtype)
                for i, img_id in enumerate(ids.tolist()):
                    key = str(int(img_id))
                    if key in h5:
                        del h5[key]
                    h5.create_dataset(
                        key, data=feats_np[i], compression="gzip", compression_opts=4
                    )
    print(f"[precompute] wrote {args.out}")


if __name__ == "__main__":
    main()
