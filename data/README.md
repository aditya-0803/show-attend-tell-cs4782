# Data

The actual image/caption files are NOT committed to git. Follow these steps to obtain them.

## 1. MS COCO 2014

Images and captions are freely available from the official COCO site (https://cocodataset.org). The files we need are:

- `train2014.zip` (~13 GB) — training images
- `val2014.zip` (~6 GB) — validation images
- `annotations_trainval2014.zip` (~250 MB) — captions and metadata

Our helper script handles this automatically:

```bash
python -m code.data.download_coco --root data/coco2014
```

After the download, the directory should look like:

```
data/coco2014/
├── annotations/
│   ├── captions_train2014.json
│   └── captions_val2014.json
├── train2014/   # image files: COCO_train2014_*.jpg
└── val2014/     # image files: COCO_val2014_*.jpg
```

## 2. Karpathy splits

Standard train / val / test splits used by the original paper and nearly all follow-up work. Download the JSON from Andrej Karpathy's site:

- http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip

Inside the ZIP, only `dataset_coco.json` is needed. Place it at `data/karpathy_splits/dataset_coco.json`.

Our helper script does this:

```bash
python -m code.data.prepare_karpathy --root data/karpathy_splits
```

The splits partition the 123k images as:

- **113,287 train**
- **5,000 val**
- **5,000 test**

## 3. Vocabulary

Built from the training captions, lower-cased, with tokens that appear fewer than 5 times (by default) mapped to `<unk>`. We keep the top 10,000 most frequent tokens, matching the paper.

```bash
python -m code.data.vocab \
    --splits data/karpathy_splits/dataset_coco.json \
    --min-count 5 --max-size 10000 --out data/vocab.json
```

This produces `data/vocab.json` with `<pad>=0`, `<start>=1`, `<end>=2`, `<unk>=3`, then word ids 4..9999.

## 4. (Optional but recommended) Precomputed VGG features

Precomputing the 14×14×512 VGG features once and caching them in an HDF5 file speeds up training by ~10x (decoder-only training, no image forward pass during every epoch).

```bash
python -m code.data.precompute_features \
    --coco-root data/coco2014 \
    --splits data/karpathy_splits/dataset_coco.json \
    --out features/coco_vgg16.h5
```

Storage: ~15–20 GB for all 123k images at fp16.
