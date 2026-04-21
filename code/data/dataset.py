"""PyTorch Datasets and DataLoaders for COCO captioning.

Two backends are provided:

* ``CocoCaptionImageDataset``  — reads raw JPEGs and applies a torchvision transform.
  Use this if you have not precomputed VGG features (slower, ~50x).

* ``CocoCaptionFeatureDataset`` — reads precomputed 14x14x512 VGG features from an
  HDF5 file built by ``code.data.precompute_features``. This is the fast path and what
  you should use for multi-epoch training on a large dataset.

Both datasets emit triples ``(features_or_image, caption_ids, caption_length)``.
A custom ``collate_fn`` pads caption_ids to the max length in the batch.

A ``LengthBucketBatchSampler`` groups samples of similar caption length into the same
batch, which both reduces padding waste and matches the proposal.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from .vocab import Vocabulary, tokenize


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

@dataclass
class CaptionSample:
    """A single image-caption training example."""
    image_id: int           # COCO image id
    file_name: str          # e.g. "train2014/COCO_train2014_000000000009.jpg"
    caption: str            # raw text
    token_ids: List[int]    # encoded (with <start>/<end>)
    split: str              # "train" | "val" | "test" | "restval"


def load_karpathy(
    karpathy_json: str | Path,
    vocab: Vocabulary,
    split: str,
    max_length: int = 22,
) -> list[CaptionSample]:
    """Load one split from the Karpathy JSON, tokenize + encode with ``vocab``.

    Following the paper we train on ``train`` + ``restval`` (~113k images);
    ``val`` / ``test`` are the Karpathy 5k / 5k evaluation sets.
    """
    with open(karpathy_json) as f:
        data = json.load(f)

    if split == "train":
        wanted = {"train", "restval"}
    elif split in {"val", "test"}:
        wanted = {split}
    else:
        raise ValueError(f"unknown split {split!r}")

    samples: list[CaptionSample] = []
    for img in data["images"]:
        if img["split"] not in wanted:
            continue
        file_name = f"{img['filepath']}/{img['filename']}"
        for sent in img["sentences"]:
            tokens = tokenize(sent["raw"])
            if not tokens:
                continue
            # Truncate to leave room for <start>/<end> within max_length.
            tokens = tokens[: max_length - 2]
            ids = vocab.encode(tokens, add_special=True)
            samples.append(
                CaptionSample(
                    image_id=img["cocoid"],
                    file_name=file_name,
                    caption=sent["raw"],
                    token_ids=ids,
                    split=img["split"],
                )
            )
    return samples


def image_transform(train: bool = False) -> transforms.Compose:
    """Standard VGG-16 input transform (224x224, ImageNet mean/std)."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# --------------------------------------------------------------------------- #
# Image-backed dataset                                                        #
# --------------------------------------------------------------------------- #

class CocoCaptionImageDataset(Dataset):
    """Reads raw images from disk; slow but doesn't require precomputation."""

    def __init__(
        self,
        samples: Sequence[CaptionSample],
        coco_root: str | Path,
        transform: Optional[Callable] = None,
    ):
        self.samples = list(samples)
        self.coco_root = Path(coco_root)
        self.transform = transform or image_transform(train=False)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        with Image.open(self.coco_root / s.file_name) as img:
            img = img.convert("RGB")
            img_tensor = self.transform(img)
        ids = torch.tensor(s.token_ids, dtype=torch.long)
        return img_tensor, ids, len(ids)


# --------------------------------------------------------------------------- #
# Feature-backed dataset (fast path)                                          #
# --------------------------------------------------------------------------- #

class CocoCaptionFeatureDataset(Dataset):
    """Reads precomputed 14x14x512 VGG features from an HDF5 file.

    The HDF5 is expected to contain one dataset per image, keyed by the string
    ``str(image_id)``, with shape ``(196, 512)`` or ``(14, 14, 512)``.
    """

    def __init__(self, samples: Sequence[CaptionSample], features_h5: str | Path):
        import h5py  # lazy import

        self.samples = list(samples)
        self._h5_path = str(features_h5)
        # Open lazily per-worker for DataLoader num_workers > 0 safety.
        self._h5 = None

    def _open(self):
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self._h5_path, "r")
        return self._h5

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        h5 = self._open()
        s = self.samples[idx]
        feats = np.asarray(h5[str(s.image_id)], dtype=np.float32)
        if feats.ndim == 3:  # (14, 14, 512) -> (196, 512)
            feats = feats.reshape(-1, feats.shape[-1])
        feats_t = torch.from_numpy(feats)
        ids = torch.tensor(s.token_ids, dtype=torch.long)
        return feats_t, ids, len(ids)


# --------------------------------------------------------------------------- #
# Batching                                                                    #
# --------------------------------------------------------------------------- #

def collate_pad(batch, pad_id: int = 0):
    """Pad token ids to the max length in the batch.

    Returns:
        features: (B, 196, 512) or (B, 3, H, W)
        caps:     (B, T) long, right-padded with ``pad_id``
        lengths:  (B,)   long
    """
    batch.sort(key=lambda x: x[2], reverse=True)
    feats, caps, lens = zip(*batch)
    feats = torch.stack(feats, dim=0)
    lengths = torch.tensor(lens, dtype=torch.long)
    T = int(lengths.max().item())
    padded = torch.full((len(caps), T), pad_id, dtype=torch.long)
    for i, c in enumerate(caps):
        padded[i, : len(c)] = c
    return feats, padded, lengths


class LengthBucketBatchSampler(Sampler[List[int]]):
    """Groups samples of similar caption length into the same batch.

    Implementation: sort all indices by caption length, then chop into contiguous
    batches of size ``batch_size`` and shuffle the order of those batches each epoch.
    This keeps intra-batch padding small while preserving randomness at the batch level.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        self.seed = seed
        # Precompute lengths once.
        self._lengths = np.array(
            [len(s.token_ids) for s in dataset.samples], dtype=np.int64
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        # Sort by length, break ties randomly to avoid always seeing the same ordering.
        order = np.argsort(
            self._lengths + rng.random() * 1e-6, kind="stable"
        ).tolist()
        # Chop into batches.
        batches = [
            order[i : i + self.batch_size]
            for i in range(0, len(order), self.batch_size)
        ]
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches = batches[:-1]
        if self.shuffle:
            rng.shuffle(batches)
        for b in batches:
            yield b

    def __len__(self) -> int:
        n = len(self._lengths)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size
