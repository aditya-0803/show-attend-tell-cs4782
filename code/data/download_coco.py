"""Download and extract MS COCO 2014 images + captions.

Usage:
    python -m code.data.download_coco --root data/coco2014
"""
from __future__ import annotations

import argparse
import os
import urllib.request
import zipfile
from pathlib import Path

COCO_URLS = {
    "train2014.zip": "http://images.cocodataset.org/zips/train2014.zip",
    "val2014.zip": "http://images.cocodataset.org/zips/val2014.zip",
    "annotations_trainval2014.zip": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
}


def _download(url: str, dst: Path) -> None:
    """Download `url` to `dst`, showing a crude progress bar."""
    if dst.exists():
        print(f"[skip] {dst} already exists")
        return
    print(f"[download] {url} -> {dst}")
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        read = 0
        chunk = 1 << 20  # 1 MiB
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            read += len(buf)
            if total:
                pct = 100 * read / total
                print(f"\r  {read / 1e9:.2f}/{total / 1e9:.2f} GB ({pct:5.1f}%)", end="")
        print()
    tmp.rename(dst)


def _unzip(zip_path: Path, extract_to: Path) -> None:
    print(f"[unzip] {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MS COCO 2014.")
    parser.add_argument("--root", type=Path, default=Path("data/coco2014"),
                        help="Destination directory.")
    parser.add_argument("--keep-zips", action="store_true",
                        help="Keep the downloaded zip files after extraction.")
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    for fname, url in COCO_URLS.items():
        zpath = args.root / fname
        _download(url, zpath)
        _unzip(zpath, args.root)
        if not args.keep_zips:
            os.remove(zpath)

    expected = [
        args.root / "annotations" / "captions_train2014.json",
        args.root / "annotations" / "captions_val2014.json",
        args.root / "train2014",
        args.root / "val2014",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise SystemExit(f"Missing after extract: {missing}")
    print(f"[done] COCO 2014 available under {args.root}")


if __name__ == "__main__":
    main()
