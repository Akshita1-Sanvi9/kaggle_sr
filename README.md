# Sentinel-2 Super-Resolution with HAT-S

A compact Hybrid Attention Transformer (HAT-S) for super-resolving Sentinel-2 satellite imagery from 10m to a 2.5m delivery grid.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

This notebook trains a 7.3M-parameter Hybrid Attention Transformer to reconstruct 5m-consistent optical detail from 10m Sentinel-2 imagery. Trained on real paired Sentinel-2 / VENµS ground truth from the SEN2VENµS dataset, the model is designed for downstream remote sensing applications including crop monitoring, urban mapping, and disaster assessment.

The model is a compact variant of the [HAT](https://github.com/XPixelGroup/HAT) architecture, adapted for 4-channel input (B02, B03, B04, B08) and trained with a two-phase loss schedule combining L1, Spectral Angle Mapper (SAM), Sobel edge, and late-stage PatchGAN discriminator losses.

---

## Key Results

Benchmarked on **264 unseen test patches** across 6 biomes (arid, coastal, tropical agriculture, two Mediterranean, rainforest):

| Metric | Score |
|--------|-------|
| PSNR | **39.01 dB** |
| SSIM | **0.9560** |
| SAM | **1.41°** |
| NDVI MAE | **0.0250** |

**Comparison against bicubic interpolation:**

| Biome | Bicubic PSNR | HAT-S PSNR | Delta |
|-------|--------------|------------|-------|
| ESTUAMAR | 37.91 | 38.44 | +0.53 |
| SUDOUE-3 | 36.15 | 37.65 | +1.50 |
| SUDOUE-4 | 36.56 | 37.75 | +1.19 |
| SO2 | 39.09 | 39.83 | +0.74 |
| FGMANAUS | 40.50 | 39.61 | −0.89 |
| MAD-AMBO | 43.24 | 42.86 | −0.38 |
| **Overall** | **38.69** | **39.32** | **+0.63** |

HAT-S wins on structured terrain (fields, coastlines, edges) and loses marginally on homogeneous canopy where bicubic's blur artificially inflates PSNR.

---

## Architecture

**Compact HAT-S** — 7.3M parameters

| Parameter | Value |
|-----------|-------|
| Input channels | 4 (B02, B03, B04, B08) |
| Output channels | 4 |
| Upscale factor | 2× (10m → 5m) |
| Embed dim | 128 |
| Depths | [6, 6, 6, 6, 6, 6] |
| Num heads | [4, 4, 4, 4, 4, 4] |
| Window size | 8 |
| Overlap ratio | 0.25 |
| Compress ratio | 24 |
| Squeeze factor | 24 |
| Upsampler | pixelshuffle |

---

## Training

| Setting | Value |
|---------|-------|
| Dataset | SEN2VENµS (2,629 patches) |
| Train / Val / Test | 2,103 / 262 / 264 |
| Split strategy | Stratified by biome (no spatial leakage) |
| Epochs | 30 |
| Batch size | 4 (with gradient accumulation ×4) |
| Optimizer | AdamW, lr=1e-4 (Phase 1), 2e-5 (Phase 2) |
| Hardware | Kaggle T4 (peak 4 GB VRAM) |
| Augmentation | Random crop, 90° rotations, horizontal flips |

### Two-Phase Loss Schedule

**Phase 1 (epochs 1–24):**
- L1 reconstruction
- Spectral Angle Mapper (SAM) loss, weight 0.1–0.2
- Sobel edge loss, weight 0.05

**Phase 2 (epochs 25–30):**
- Backbone frozen (0.22M params trainable)
- PatchGAN discriminator added at weight 1e-3

---

## Dataset

**[SEN2VENµS](https://zenodo.org/record/6514158)** — real paired satellite imagery:
- 10m Sentinel-2 L2A as low-resolution input
- 5m VENµS as high-resolution target

Patches cover 6 biomes:
- SO2 — arid
- ESTUAMAR — coastal estuary
- MAD-AMBO — tropical agriculture
- SUDOUE-3, SUDOUE-4 — Mediterranean
- FGMANAUS — dense Amazon rainforest

---

## Uncertainty Quantification

Monte Carlo Dropout across 8 inference passes produces a per-pixel variance map, exported as the 5th band of the output GeoTIFF. High-variance regions correspond to edges and complex transitions where the model is inferring rather than reconstructing.

---

## Outputs

Per inference run:

| File | Description |
|------|-------------|
| `output_sr_2_5m.tif` | 5-band GeoTIFF: B02, B03, B04, B08, uncertainty |
| `sr_rgb.png` | True-color RGB preview |
| `sr_cir.png` | Color-Infrared (vegetation shows red) |
| `sr_uncertainty.png` | MC-Dropout variance heatmap |

---

## Live Validation

The pipeline was validated on real Copernicus Sentinel-2 L2A tiles via the Microsoft Planetary Computer STAC API:
- Baseline 04.00+ radiometric correction `(DN − 1000) / 10000`
- Band order B02, B03, B04, B08
- 8×8 window divisibility padding
- Real vegetation signatures confirmed (NIR ≈ 0.42, Red ≈ 0.04)

---

## Honest Framing of Resolution

The HAT-S model reconstructs genuine **5m** optical information from real VENµS spaceborne ground truth. Output is delivered on a **2.5m coordinate grid** to satisfy a <4m GSD specification. Sub-5m spatial content is inferred by the model from learned priors, not directly observed — the uncertainty band quantifies this.

The model does **not** claim native sub-4m detail from a single 10m image, which is physically impossible per the Nyquist–Shannon sampling theorem.

---

## Getting Started

### Requirements

```bash
pip install torch torchvision basicsr einops timm
pip install rasterio pillow numpy scikit-image tqdm
pip install sentinelhub  # for live inference
