"""Recompute METEOR on the existing hyps in test_metrics.json using the
paper-comparable Java METEOR implementation (via the meteor-1.5.jar shipped
with pycocoevalcap, but in batch mode — bypassing the buggy streaming protocol).

Usage:
    python -m code.recompute_meteor \\
        --metrics-json /path/to/test_metrics.json \\
        --splits       /path/to/dataset_coco.json \\
        --split        test
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .utils.metrics import compute_meteor_java_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics-json", type=Path, required=True)
    p.add_argument("--splits",       type=Path, required=True)
    p.add_argument("--split",        type=str, default="test")
    args = p.parse_args()

    with open(args.metrics_json) as f:
        data = json.load(f)
    hyps_raw = data.get("hyps", {})
    if not hyps_raw:
        raise SystemExit("[err] no 'hyps' field in metrics-json")

    with open(args.splits) as f:
        ds = json.load(f)
    refs = defaultdict(list)
    for img in ds["images"]:
        if img["split"] != args.split:
            continue
        for s in img["sentences"]:
            refs[img["cocoid"]].append(s["raw"])

    # Align keys (hyps keys may be strings).
    hyps = {int(k): v for k, v in hyps_raw.items() if int(k) in refs}
    refs = {k: refs[k] for k in hyps}
    print(f"[rescore] scoring {len(hyps)} hypotheses against Karpathy '{args.split}' refs")

    score = compute_meteor_java_batch(refs, hyps)
    print(f"[rescore] METEOR (Java, paper scale) = {score:.4f}")

    # Patch the file: keep NLTK score under METEOR_NLTK for posterity, write Java
    # score as METEOR.
    nltk_meteor = data["scores"].get("METEOR")
    data["scores"]["METEOR_NLTK"] = nltk_meteor
    data["scores"]["METEOR"] = score
    with open(args.metrics_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[rescore] patched {args.metrics_json}")
    print(f"          METEOR (Java) = {score:.4f}")
    print(f"          METEOR (NLTK) = {nltk_meteor:.4f}  (kept as METEOR_NLTK)")


if __name__ == "__main__":
    main()
