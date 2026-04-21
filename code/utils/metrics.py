"""BLEU-1..4 and METEOR computation.

We use ``pycocoevalcap`` (standard MS COCO evaluation suite) if available, because it
matches what most captioning papers report. For BLEU, the paper states "without
brevity penalty"; ``pycocoevalcap``'s BLEU implementation includes the standard brevity
penalty, so we also provide an NLTK-based BLEU fallback that lets us set
``smoothing_function`` and disable the BP if needed.

Typical usage:

    refs = {img_id: [ref_str1, ref_str2, ...], ...}
    hyps = {img_id: hyp_str, ...}
    scores = compute_all_metrics(refs, hyps)
    # -> {"BLEU-1": ..., "BLEU-2": ..., "BLEU-3": ..., "BLEU-4": ..., "METEOR": ...}
"""
from __future__ import annotations

from typing import Dict, List, Sequence


def _normalize_refs_hyps(refs: Dict, hyps: Dict):
    """Make sure keys line up and values are in the expected format for pycocoevalcap."""
    # Wrap hypotheses as single-element lists (pycocoevalcap expects {id: [str]}).
    gts = {k: list(v) for k, v in refs.items()}
    res = {k: [v] if isinstance(v, str) else list(v) for k, v in hyps.items()}
    common = sorted(set(gts) & set(res))
    gts = {k: gts[k] for k in common}
    res = {k: res[k] for k in common}
    return gts, res


def compute_bleu_coco(refs: Dict, hyps: Dict) -> Dict[str, float]:
    """BLEU-1..4 using the official MS COCO eval code (with brevity penalty)."""
    from pycocoevalcap.bleu.bleu import Bleu
    gts, res = _normalize_refs_hyps(refs, hyps)
    score, _ = Bleu(4).compute_score(gts, res)
    return {f"BLEU-{i+1}": float(score[i]) * 100.0 for i in range(4)}


def compute_bleu_nltk(
    refs: Dict, hyps: Dict, brevity_penalty: bool = False
) -> Dict[str, float]:
    """BLEU-1..4 using NLTK's corpus_bleu.

    The paper reports scores "without brevity penalty"; set ``brevity_penalty=False`` to
    match that convention. We smooth using ``SmoothingFunction().method1``, which is
    standard practice for caption-level BLEU (short sentences otherwise collapse to 0).
    """
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    gts, res = _normalize_refs_hyps(refs, hyps)
    list_of_refs: list[list[list[str]]] = []
    list_of_hyps: list[list[str]] = []
    for k in gts:
        list_of_refs.append([r.lower().split() for r in gts[k]])
        list_of_hyps.append(res[k][0].lower().split())

    sm = SmoothingFunction().method1
    weights_sets = [
        (1.0, 0.0, 0.0, 0.0),
        (0.5, 0.5, 0.0, 0.0),
        (1 / 3, 1 / 3, 1 / 3, 0.0),
        (0.25, 0.25, 0.25, 0.25),
    ]
    scores: Dict[str, float] = {}
    for i, w in enumerate(weights_sets, start=1):
        s = corpus_bleu(
            list_of_refs,
            list_of_hyps,
            weights=w,
            smoothing_function=sm,
            auto_reweigh=False,
        )
        # "Without brevity penalty" means multiplying out the BP when it's < 1.
        if not brevity_penalty:
            # Recompute without BP: BLEU = exp(sum w_n * log p_n). We can do that by
            # calling modified_precision directly; but for simplicity we scale s by
            # 1 / BP.
            from nltk.translate.bleu_score import closest_ref_length
            from math import exp

            hyp_len = sum(len(h) for h in list_of_hyps)
            ref_len = sum(
                closest_ref_length(rs, len(h))
                for rs, h in zip(list_of_refs, list_of_hyps)
            )
            if hyp_len > 0 and hyp_len < ref_len:
                bp = exp(1 - ref_len / hyp_len)
                s = s / max(bp, 1e-12)
        scores[f"BLEU-{i}"] = float(s) * 100.0
    return scores


def _ensure_nltk_data() -> None:
    """Make sure the NLTK corpora that METEOR needs are available."""
    import nltk
    needed = [
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("tokenizers/punkt", "punkt"),
    ]
    for resource, pkg in needed:
        try:
            nltk.data.find(resource)
        except LookupError:
            nltk.download(pkg, quiet=True)


def compute_meteor_nltk(refs: Dict, hyps: Dict) -> float:
    """METEOR via NLTK (pure Python — no Java, no JAR).

    NLTK's ``meteor_score`` computes a sentence-level score given tokenized references
    and a tokenized hypothesis. We average over the corpus for the final number, which
    is the standard interpretation and what pycocoevalcap also does.
    """
    _ensure_nltk_data()
    from nltk.translate.meteor_score import meteor_score

    gts, res = _normalize_refs_hyps(refs, hyps)
    total, count = 0.0, 0
    for k in gts:
        ref_tokens = [r.lower().split() for r in gts[k]]
        hyp_tokens = res[k][0].lower().split()
        if not hyp_tokens:
            continue
        try:
            total += meteor_score(ref_tokens, hyp_tokens)
            count += 1
        except Exception:
            # Rare edge cases (empty ref, unusual tokens) — skip silently.
            continue
    if count == 0:
        return 0.0
    return (total / count) * 100.0


def compute_meteor_coco(refs: Dict, hyps: Dict) -> float:
    """METEOR via pycocoevalcap (requires Java). Kept as a fallback."""
    from pycocoevalcap.meteor.meteor import Meteor
    gts, res = _normalize_refs_hyps(refs, hyps)
    score, _ = Meteor().compute_score(gts, res)
    return float(score) * 100.0


def compute_meteor(refs: Dict, hyps: Dict) -> float:
    """METEOR — prefers NLTK's pure-Python implementation; falls back to pycocoevalcap."""
    try:
        return compute_meteor_nltk(refs, hyps)
    except Exception as e:
        print(f"[metrics] NLTK METEOR failed ({e}); trying pycocoevalcap")
        return compute_meteor_coco(refs, hyps)


def compute_all_metrics(
    refs: Dict,
    hyps: Dict,
    bleu_brevity_penalty: bool = False,
) -> Dict[str, float]:
    """Return BLEU-1..4 + METEOR in a single dict, tolerating missing optional deps.

    ``bleu_brevity_penalty=False`` reproduces the convention used in the paper.
    """
    results: Dict[str, float] = {}
    try:
        results.update(
            compute_bleu_nltk(refs, hyps, brevity_penalty=bleu_brevity_penalty)
        )
    except Exception as e:
        print(f"[metrics] NLTK BLEU failed: {e}; falling back to COCO BLEU")
        results.update(compute_bleu_coco(refs, hyps))

    try:
        results["METEOR"] = compute_meteor(refs, hyps)
    except Exception as e:
        print(f"[metrics] METEOR unavailable (need Java + pycocoevalcap): {e}")

    return results


# Back-compat aliases expected by __init__.
compute_bleu = compute_bleu_nltk
