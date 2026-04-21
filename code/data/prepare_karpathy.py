"""Download the Karpathy splits JSON (COCO) used by the paper and most follow-up work.

Usage:
    python -m code.data.prepare_karpathy --root data/karpathy_splits
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

KARPATHY_URL = "http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"


def _download(url: str, dst: Path) -> None:
    if dst.exists():
        print(f"[skip] {dst} already exists")
        return
    print(f"[download] {url}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/karpathy_splits"))
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    zpath = args.root / "caption_datasets.zip"
    _download(KARPATHY_URL, zpath)

    # The zip contains dataset_coco.json, dataset_flickr8k.json, dataset_flickr30k.json.
    target = args.root / "dataset_coco.json"
    if not target.exists():
        with zipfile.ZipFile(zpath) as zf:
            with zf.open("dataset_coco.json") as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    zpath.unlink(missing_ok=True)
    print(f"[done] wrote {target}")


if __name__ == "__main__":
    main()
