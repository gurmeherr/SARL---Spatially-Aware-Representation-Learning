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
