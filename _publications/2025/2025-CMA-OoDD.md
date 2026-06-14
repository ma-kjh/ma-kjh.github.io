---
title:          "Enhanced OoD Detection through Cross-Modal Alignment of Multi-Modal Representations"
date:           2025-03-24 00:01:00 +0800
selected:       true
pub:            "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"
# pub_pre:        "Submitted to "
# pub_post:       'Under review.'
# pub_last:       ' <span class="badge badge-pill badge-publication badge-success">Oral</span> <span class="badge badge-warning badge-publication badge-danger">Best Paper Award</span>'
pub_date:       "2025"
abstract: >-
  We show that multi-modal fine-tuning (MMFT) of vision-language models like CLIP can achieve strong out-of-distribution detection (OoDD), and identify the modality gap in in-distribution embeddings as a key limitation of naïve fine-tuning. To address this, we propose a training objective that enhances cross-modal alignment by regularizing the distances between image and text embeddings, which we theoretically show corresponds to maximum likelihood estimation of an energy-based model on a hypersphere. Combined with post-hoc methods such as NegLabel, our approach achieves state-of-the-art OoDD performance and leading ID accuracy on ImageNet-1k benchmarks.
cover:          /assets/images/covers/CMA.png
authors:
  - <strong>Jeonghyeon Kim</strong>
  - Sangheum Hwang
links:
  Paper: https://arxiv.org/abs/2503.18817
  Code : https://github.com/ma-kjh/CMA-OoDD
---
