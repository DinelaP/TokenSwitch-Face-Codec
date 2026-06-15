# Switchable Token-Specific Codebook Quantization for Face Image Compression

**MMS 2025/26 — Image Compression Methods Project**  
Faculty of Electrical Engineering, University of Sarajevo  
Course: Multimedia Systems (MMS)

---

## Overview

This repository contains the implementation of the paper:

> **Switchable Token-Specific Codebook Quantization For Face Image Compression**  
> Yongbo Wang, Haonan Wang, Guodong Mu, et al.  
> *NeurIPS 2025*  
> [Paper (OpenReview)](https://openreview.net/pdf?id=8AtYSW3VdX)

The method addresses a key bottleneck in VQ-VAE-style image compression: all tokens share a single global codebook. This work introduces a **hierarchical, switchable codebook** mechanism that assigns distinct codebook groups per image category and an independent sub-codebook per token, improving reconstruction quality at ultra-low bitrates (bpp).

---

## Method Summary

Standard codebook-based compression uses a single shared codebook of size *N*, costing `T × ⌈log₂N⌉` bits for *T* tokens. This work replaces it with:

- **Image-level routing** — a routing module selects one of *M* codebook groups based on image category/attributes (e.g., gender, age, ethnicity in face images).
- **Token-level assignment** — within the selected group, each of the *T* tokens is assigned its own sub-codebook of size *K*.
- **Reduced storage cost** — `T × ⌈log₂K⌉ + ⌈log₂M⌉` bits, enabling larger total codebook capacity at the same or lower bpp.

The module is **plug-and-play** and can be integrated into any existing codebook-based compression framework (e.g., TiTok, VQ-VAE).


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
| `hyperprior-qN`   | Ballé et al. **2018** (main)  | 1, 3, 5, 8     |
| `mbt2018-qN`      | Minnen et al. 2018            | 3, 5, 8        |
| `cheng2020-qN`    | Cheng et al. 2020             | 3, 6           |

---

## Metrics

| Metric   | Range      | Better | Meaning                                      |
|----------|------------|--------|----------------------------------------------|
| PSNR     | 20–50 dB   | Higher | Pixel-level reconstruction fidelity          |
| MS-SSIM  | 0–1        | Higher | Perceptual quality (luminance, contrast, structure) |
| bpp      | 0.1–2.0    | Lower  | Compression ratio (bits per pixel)           |
| BD-Rate  | % (±)      | Lower  | Bit-rate saving vs. anchor at same quality   |

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
