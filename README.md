# Switchable Token-Specific Codebook Quantization for Face Image Compression

**MMS 2025/26 — Image Compression Methods**  
Faculty of Electrical Engineering, University of Sarajevo

---

This project is part of the MMS 2025/26 course assignment on image compression methods. We implement and evaluate neural image compression models on the ImageNet-1k dataset, based on the paper:

> **Switchable Token-Specific Codebook Quantization For Face Image Compression**  
> Wang et al., NeurIPS 2025 · [Paper](https://openreview.net/pdf?id=8AtYSW3VdX) · [arXiv](https://arxiv.org/abs/2510.22943)

The paper proposes a hierarchical codebook quantization mechanism that assigns distinct codebook groups per image category and an independent sub-codebook per token, significantly improving reconstruction quality at ultra-low bitrates compared to standard VQ-VAE approaches. Since the authors did not release official code, this implementation uses [`compressai`](https://github.com/InterDigitalInc/CompressAI) baseline models as comparison points and evaluates them on ImageNet-1k val using the same metrics as the paper.

---

## Repository Structure

```
compress_imagenet/
├── evaluate.py          # Main evaluation script
├── colab_notebook.ipynb # Colab-ready notebook (recommended)
├── requirements.txt
└── README.md
```

---

## Quick Start (Google Colab — Recommended)

1. Open `colab_notebook.ipynb` in Google Colab
2. Set Runtime → T4 GPU
3. Run all cells top to bottom (~20-30 min)

---

## Local Setup

```bash
pip install -r requirements.txt

python evaluate.py \
  --data-dir /path/to/imagenet/val \
  --num-images 500 \
  --patch-size 256 \
  --batch-size 8 \
  --save-examples \
  --output-json results.json
```

### ImageNet val folder structure expected:
```
imagenet_val/
  n01440764/   ← class folder (any name works)
    img1.JPEG
    img2.JPEG
  n01443537/
    ...
```

---

## Models Evaluated

| Model Name        | Paper                         | Quality Levels |
|-------------------|-------------------------------|----------------|
| `factorized-qN`   | Ballé et al. 2017 (baseline)  | 1, 3, 5, 8     |
| `hyperprior-qN`   | Ballé et al. 2018 (main)      | 1, 3, 5, 8     |
| `mbt2018-qN`      | Minnen et al. 2018            | 3, 5, 8        |
| `cheng2020-qN`    | Cheng et al. 2020             | 3, 6           |

---

## Metrics

| Metric   | Range      | Better | Meaning                                           |
|----------|------------|--------|---------------------------------------------------|
| PSNR     | 20–50 dB   | Higher | Pixel-level reconstruction fidelity               |
| MS-SSIM  | 0–1        | Higher | Perceptual quality (luminance, contrast, structure)|
| bpp      | 0.1–2.0    | Lower  | Compression ratio (bits per pixel)                |
| BD-Rate  | % (±)      | Lower  | Bit-rate saving vs. anchor at same quality        |

---

## Expected Results (approximate, 256×256 crops, ImageNet-1k val)

| Model           | bpp   | PSNR (dB) | MS-SSIM |
|-----------------|-------|-----------|---------|
| factorized-q1   | 0.13  | 28.5      | 0.912   |
| factorized-q5   | 0.45  | 33.1      | 0.962   |
| hyperprior-q1   | 0.12  | 29.2      | 0.921   |
| hyperprior-q5   | 0.42  | 34.1      | 0.971   |
| cheng2020-q6    | 0.80  | 37.5      | 0.985   |

---

## Requirements

```
torch>=2.0
torchvision>=0.15
compressai>=1.2
pytorch-msssim>=0.2
scipy>=1.10
numpy
Pillow
matplotlib
```

---

**Supervisor:** vhasic1@etf.unsa.ba  
