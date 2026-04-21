# Project Roadmap

**Course:** CS 4782 — Deep Learning, Spring 2026
**Team:** Aditya Iyer (aji8), Aitzaz Shaikh (ams845), Joshua George (jjg322), Tailai Ying (tty6)
**Paper:** *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention* — Xu et al., ICML 2015
**Target:** Reproduce the **soft attention** MS COCO numbers from Table 1: BLEU-1 70.7 / BLEU-2 49.2 / BLEU-3 34.4 / BLEU-4 24.3 / METEOR 23.90

---

## Deadlines

| Deliverable | Due | Submit To |
|---|---|---|
| Poster (PDF) | **April 30, 2026** | GitHub `poster/` + printed copy in class |
| 2-page Report | May 12, 2026 | Gradescope + GitHub `report/` |
| GitHub Repo | May 12, 2026 | Gradescope |

---

## Day-by-day plan

### Week of April 20 — Build & first training run

- **Mon Apr 20 (done):** Proposal approved.
- **Tue Apr 21:** Repo scaffold, full code written (data / model / training / eval / beam / viz), Colab notebook ready. *[TODAY]*
- **Wed Apr 22 morning:** Team reads through code, raises issues. Kick off feature-precomputation job on Colab (VGG-16 fc features for the full Karpathy train+val+test set — ~15–20 GB of cached tensors but cuts per-epoch time by 10x).
- **Wed Apr 22 evening:** Start first full training run (up to 50 epochs or 12 hrs, whichever comes first). Log to TensorBoard.
- **Thu Apr 23 – Fri Apr 24:** Monitor training. If val BLEU-4 plateaus early, tweak LR decay / dropout / λ for doubly-stochastic regularization. Begin drafting poster layout.
- **Sat Apr 25 – Sun Apr 26:** Second training run if needed. Generate attention-visualization images on 8–10 test images.
- **Mon Apr 27:** Freeze numbers. Final attention figures.
- **Tue Apr 28:** First full poster draft.
- **Wed Apr 29:** Revise poster, print it (Cornell ITS or Print & Copy in the Cornell Store — budget 24 hrs for delivery).
- **Thu Apr 30:** **Poster presentation.**

### Week of May 4 — Report & polish

- **May 4–8:** Draft 2-page report (methodology + results + reflections are the weighted sections per the rubric).
- **May 9–10:** Polish README (TL;DR of the report), write reproduction steps.
- **May 11:** Final review pass.
- **May 12:** Submit final repo + report.

---

## Team split

| Person | Primary ownership | Backup |
|---|---|---|
| Adi | Training loop, evaluation harness, overall code coherence | Poster |
| Aitzaz | Data pipeline (COCO download, Karpathy splits, vocab, Dataset) | Report |
| Joshua | Model modules (encoder / attention / decoder) | Beam search |
| Tailai | Beam search, attention visualizations, poster design | Eval |

Report is shared (each person writes the section they know best); Adi does the final pass.

---

## Risk register

1. **Training is slower than planned.** Mitigation: precompute VGG features once; train decoder-only. A single epoch over 110k Karpathy-train captions on cached features should be ~5–10 minutes on A100.
2. **Metrics fall short of paper.** Mitigation: acceptable per the instructions — "A failure to match the reported results could happen for any number of reasons that would not negatively impact your grade." What matters is a clear reflection on *why*.
3. **Colab disconnects mid-training.** Mitigation: checkpoint every epoch to Google Drive; resume from last checkpoint.
4. **COCO download is slow.** Mitigation: use the official `cocodataset.org` mirrors; the images are ~20 GB but Colab downloads are fast. Do this once into Drive and reuse.
5. **Poster print deadline.** Mitigation: target Apr 28 for a print-ready draft so there's a full day of buffer.

---

## What's already decided (from the proposal)

- **Encoder:** frozen VGG-16, conv5_3 pre-pool features (14×14×512 → 196 vectors of dim 512)
- **Attention:** two-layer MLP, softmax over 196 regions, doubly-stochastic regularization λ=1
- **Decoder:** 1-layer LSTM, hidden=512, embed=512, init h/c from mean annotation (separate linear projections), deep output layer (linear + tanh) over 10k vocab
- **Training:** Adam lr=4e-4 with 0.8× decay every 5 epochs, batch 32 bucketed by caption length, dropout 0.5, up to 50 epochs, early stop on val BLEU-4
- **Metrics:** BLEU-1..4 (no brevity penalty, per the paper) + METEOR (standard)
- **Data:** MS COCO 2014 + Karpathy splits; 10k most-frequent-word vocabulary
