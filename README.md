# SARL: Spatially-Aware Self-Supervised Representation Learning for Visuo-Tactile Perception

Official PyTorch/Lightning implementation of **SARL: Spatially-Aware Self-Supervised Representation Learning for Visuo-Tactile Perception**.

SARL is a self-supervised learning framework for fused visuo-tactile images. It extends a BYOL-style joint-embedding architecture with three spatially-aware auxiliary objectives:

- **SAL** — Saliency Alignment
- **PPDA** — Patch-Prototype Distribution Alignment
- **RAM** — Region Affinity Matching

These losses operate on intermediate encoder feature maps and encourage the learned representation to preserve attentional focus, semantic part composition, and geometric relationships across augmented views.

---

## Overview

Most self-supervised learning methods compress an image into a single global embedding. While this works well for classification, it can discard spatial and geometric information that is important for robotic manipulation.

SARL addresses this by combining:

1. A **global BYOL objective** on final embeddings.
2. **SAL** on encoder layers 2, 3, and 4 to align salient spatial regions.
3. **PPDA** on encoder layer 3 to align local prototype-part distributions.
4. **RAM** on encoder layer 3 to preserve region-to-region geometric relationships.

The model is designed for fused visual-tactile data, especially data from vision-based tactile sensors where visual and tactile cues are spatially aligned in the same image.

---

## Method Summary

Given an input image `x`, two augmented views are generated:

```text
x1 = t(x)
x2 = t'(x)

These are passed through an online branch and a target branch.

x1, x2
  ↓
Online encoder fθ + projector gθ + predictor qθ
Target encoder fξ + projector gξ

The target branch is updated using exponential moving average:

ξ ← μξ + (1 - μ)θ

The final SARL objective is:

L_SARL = L_global + λ_SAL L_SAL + λ_PPDA L_PPDA + λ_RAM L_RAM
```


## Spatial Objectives

**1. SAL: Saliency Alignment**
SAL answers the question: Where should the model look?

It encourages the model to focus on the same salient physical regions across augmented views.

For each intermediate feature map, a saliency map is computed as the channel-wise absolute activation sum:
```
S(x) = Σc |F_c(x)|
```
The saliency maps are L2-normalized and compared across the valid overlapping regions of the two augmented views. In this implementation, crop and horizontal flip metadata are tracked so that cross-view spatial correspondence can be estimated.
