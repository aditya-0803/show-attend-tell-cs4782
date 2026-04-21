"""Vocabulary construction.

We follow the paper and use a fixed vocabulary of the 10,000 most frequent tokens in the
training captions (Karpathy split). Tokens appearing fewer than `min_count` times in
training are never added. Four special tokens are reserved:

    <pad>    = 0
    <start>  = 1
    <end>    = 2
    <unk>    = 3

Usage (CLI):
    python -m code.data.vocab \\
        --splits data/karpathy_splits/dataset_coco.json \\
        --min-count 5 --max-size 10000 --out data/vocab.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
SPECIALS = [PAD, START, END, UNK]

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(caption: str) -> List[str]:
    """Lowercase + simple regex tokenizer.

    Matches runs of alphanumeric characters, optionally joined by a single apostrophe
    (so contractions like "don't" stay as one token). Strips punctuation.
    """
    return _TOKEN_RE.findall(caption.lower())


class Vocabulary:
    """Bidirectional word <-> id mapping."""

    def __init__(self, word2id: dict[str, int]):
        self.word2id = dict(word2id)
        self.id2word = {i: w for w, i in word2id.items()}
        for tok in SPECIALS:
            assert tok in self.word2id, f"missing special token {tok}"
        self.pad_id = self.word2id[PAD]
        self.start_id = self.word2id[START]
        self.end_id = self.word2id[END]
        self.unk_id = self.word2id[UNK]

    def __len__(self) -> int:
        return len(self.word2id)

    def encode(self, tokens: Iterable[str], add_special: bool = True) -> List[int]:
        ids: List[int] = []
        if add_special:
            ids.append(self.start_id)
        for t in tokens:
            ids.append(self.word2id.get(t, self.unk_id))
        if add_special:
            ids.append(self.end_id)
        return ids

    def decode(self, ids: Iterable[int], strip_special: bool = True) -> List[str]:
        out: List[str] = []
        for i in ids:
            w = self.id2word.get(int(i), UNK)
            if strip_special and w in (PAD, START, END):
                if w == END:
                    break
                continue
            out.append(w)
        return out

    # ---- (de)serialization ----

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"word2id": self.word2id}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with open(path) as f:
            data = json.load(f)
        return cls(data["word2id"])


def build_vocab(
    karpathy_json: str | Path,
    min_count: int = 5,
    max_size: int = 10_000,
) -> Vocabulary:
    """Build a Vocabulary from Karpathy-split COCO training captions."""
    with open(karpathy_json) as f:
        dataset = json.load(f)

    counter: Counter[str] = Counter()
    for img in dataset["images"]:
        if img["split"] not in ("train", "restval"):
            continue
        for sent in img["sentences"]:
            # Karpathy already provides a tokens list per sentence; we still run our
            # tokenizer on the raw string so behavior is consistent at inference time.
            counter.update(tokenize(sent["raw"]))

    # Determine how many non-special slots are available.
    kept_slots = max_size - len(SPECIALS)
    most_common = [w for w, c in counter.most_common() if c >= min_count]
    kept_words = most_common[:kept_slots]

    word2id: dict[str, int] = {tok: i for i, tok in enumerate(SPECIALS)}
    for w in kept_words:
        word2id[w] = len(word2id)

    vocab = Vocabulary(word2id)
    print(
        f"[vocab] built with {len(vocab)} tokens "
        f"(specials={len(SPECIALS)}, kept={len(kept_words)}, "
        f"candidates>= {min_count} = {len(most_common)})"
    )
    return vocab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, required=True,
                        help="Path to Karpathy dataset_coco.json")
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--max-size", type=int, default=10_000)
    parser.add_argument("--out", type=Path, default=Path("data/vocab.json"))
    args = parser.parse_args()

    vocab = build_vocab(args.splits, min_count=args.min_count, max_size=args.max_size)
    vocab.save(args.out)
    print(f"[vocab] wrote {args.out}")


if __name__ == "__main__":
    main()
