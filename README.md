# Show, Attend and Tell — Re-implementation (CS 4782, Cornell, Spring 2026)

A PyTorch re-implementation of the **soft attention** variant of Xu et al., *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention* (ICML 2015), trained on MS COCO with the Karpathy splits.

> **Status:** in progress (final repo and results will be finalized by May 12, 2026).

## 1. Introduction

This repository is a course project for CS 4782 at Cornell. We re-implement the soft-attention image-captioning model of Xu et al. (2015) and evaluate it on MS COCO 2014, aiming to reproduce the BLEU/METEOR numbers reported in Table 1 of the paper. The model combines a frozen VGG-16 CNN encoder with an LSTM decoder that, at every time step, computes a soft attention over a 14×14 grid of image features to decide which region to "look at" when generating the next word.

## 2. Chosen Result

We reproduce the **soft attention** MS COCO row from Table 1 of the paper:

| Metric   | Paper | Ours |
|----------|-------|------|
| BLEU-1   | 70.7  | TBD  |
| BLEU-2   | 49.2  | TBD  |
| BLEU-3   | 34.4  | TBD  |
| BLEU-4   | 24.3  | TBD  |
| METEOR   | 23.90 | TBD  |

We chose soft attention (not hard) because it trains with standard backprop and is substantially more stable to reproduce than the REINFORCE-trained hard variant.

## 3. GitHub Contents

```
.
├── code/               # PyTorch implementation
│   ├── data/           # COCO download, Karpathy splits, vocab, Dataset
│   ├── models/         # encoder, attention, decoder, full captioner
│   ├── utils/          # helpers (checkpointing, logging, metrics)
│   ├── train.py        # training entry point
│   ├── evaluate.py     # BLEU-1..4 + METEOR evaluation
│   ├── generate.py     # beam-search + greedy caption generation
│   └── visualize_attention.py   # overlay alpha maps on images
├── data/               # download instructions (datasets NOT in git)
├── notebooks/          # Colab training notebook
├── results/            # metrics, attention figures, sample captions
├── poster/             # poster PDF
├── report/             # 2-page report PDF
├── requirements.txt
├── LICENSE             # MIT
└── .gitignore
```

## 4. Re-implementation Details

- **Encoder:** frozen `torchvision` VGG-16, features taken from the last conv layer before max-pool (`features[:29]` → `conv5_3 + ReLU`), producing a 14×14×512 tensor that we flatten into 196 annotation vectors of dim 512.
- **Attention:** two-layer MLP `f_att(h_{t-1}, a_i)` over each annotation vector, followed by softmax, with doubly-stochastic regularization (λ=1, Eq. 14).
- **Decoder:** single-layer LSTM, hidden=512, embedding=512, `h_0 / c_0` initialized from the mean annotation vector via separate linear projections, deep output layer (linear + tanh) over the 10k-word vocabulary (Eq. 7).
- **Training:** Adam, lr=4e-4 with 0.8× decay every 5 epochs, batch 32 grouped by caption length, dropout=0.5, up to 50 epochs, early stopping on validation BLEU-4.
- **Evaluation:** BLEU-1..4 (no brevity penalty, matching the paper) and METEOR, both via the standard `pycocoevalcap` implementation.

## 5. Reproduction Steps

```bash
# 1. Clone and install
git clone https://github.com/<user>/show-attend-tell-cs4782.git
cd show-attend-tell-cs4782
pip install -r requirements.txt

# 2. Download data (see data/README.md for details)
python -m code.data.download_coco --root data/coco2014
python -m code.data.prepare_karpathy --root data/karpathy_splits

# 3. Build vocabulary (writes data/vocab.json)
python -m code.data.vocab --splits data/karpathy_splits/dataset_coco.json \
    --min-count 5 --max-size 10000 --out data/vocab.json

# 4. (Recommended) Precompute VGG features to disk
python -m code.data.precompute_features \
    --coco-root data/coco2014 \
    --splits data/karpathy_splits/dataset_coco.json \
    --out features/coco_vgg16.h5

# 5. Train
python -m code.train \
    --features features/coco_vgg16.h5 \
    --splits data/karpathy_splits/dataset_coco.json \
    --vocab data/vocab.json \
    --out-dir checkpoints/run1

# 6. Evaluate
python -m code.evaluate \
    --checkpoint checkpoints/run1/best.pt \
    --features features/coco_vgg16.h5 \
    --vocab data/vocab.json \
    --splits data/karpathy_splits/dataset_coco.json \
    --beam 3

# 7. Generate attention visualizations
python -m code.visualize_attention \
    --checkpoint checkpoints/run1/best.pt \
    --image path/to/image.jpg \
    --vocab data/vocab.json --beam 3
```

**Compute:** we trained on an NVIDIA A100 (Google Colab Pro). Precomputing features takes ~30–45 minutes; each training epoch over cached features is ~5–10 minutes. End-to-end reproduction fits inside a single Colab Pro session.

## 6. Results / Insights

Final metrics, attention visualizations, and sample captions will be placed in `results/`. See `report/group_showattendtell_2page_report.pdf` for discussion.

## 7. Conclusion

TBD — will be written after final results are collected.

## 8. References

- Xu, K., Ba, J. L., Kiros, R., Cho, K., Courville, A., Salakhutdinov, R., Zemel, R., & Bengio, Y. (2015). *Show, Attend and Tell: Neural Image Caption Generation with Visual Attention.* ICML.
- Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). *Microsoft COCO: Common Objects in Context.* ECCV.
- Karpathy, A., & Fei-Fei, L. (2015). *Deep Visual-Semantic Alignments for Generating Image Descriptions.* CVPR.
- Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition* (VGG). arXiv:1409.1556.
- Paszke et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS.
- `pycocoevalcap` — https://github.com/salaniz/pycocoevalcap
- NLTK — https://www.nltk.org

## 9. Acknowledgements

This project was completed as part of CS 4782 (Deep Learning) at Cornell University, Spring 2026, under the guidance of the course staff. We thank them for the problem setup, feedback on the proposal, and compute access.
