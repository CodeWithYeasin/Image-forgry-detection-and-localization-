"""Evaluation metrics for forgery detection and localization."""

from __future__ import annotations

import numpy as np
import torch


def pixel_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute pixel-level Intersection over Union (IoU).

    Args:
        pred_mask: Raw logit or probability map, shape ``(B, 1, H, W)`` or ``(H, W)``.
        gt_mask: Binary ground-truth mask with the same spatial dimensions.
        threshold: Decision threshold applied to ``pred_mask``.

    Returns:
        Mean IoU across the batch.
    """
    pred_bin = (torch.sigmoid(pred_mask) > threshold).float()
    gt = gt_mask.float()

    intersection = (pred_bin * gt).sum(dim=(-2, -1))
    union = (pred_bin + gt).clamp(0, 1).sum(dim=(-2, -1))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()


def pixel_f1(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute pixel-level F1 score.

    Args:
        pred_mask: Raw logit or probability map.
        gt_mask: Binary ground-truth mask.
        threshold: Decision threshold.

    Returns:
        Mean F1 across the batch.
    """
    pred_bin = (torch.sigmoid(pred_mask) > threshold).float()
    gt = gt_mask.float()

    tp = (pred_bin * gt).sum(dim=(-2, -1))
    fp = (pred_bin * (1 - gt)).sum(dim=(-2, -1))
    fn = ((1 - pred_bin) * gt).sum(dim=(-2, -1))

    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return f1.mean().item()


def compute_metrics(
    logits: torch.Tensor,
    pred_mask: torch.Tensor,
    labels: torch.Tensor,
    gt_mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Aggregate all metrics into a single dictionary.

    Args:
        logits: Classification logits, shape ``(B, num_classes)``.
        pred_mask: Segmentation output, shape ``(B, 1, H, W)``.
        labels: Ground-truth class labels, shape ``(B,)``.
        gt_mask: Ground-truth binary masks, shape ``(B, H, W)``.
        threshold: Mask binarization threshold.

    Returns:
        Dictionary with keys ``cls_acc``, ``pixel_iou``, ``pixel_f1``.
    """
    preds = logits.argmax(dim=1)
    cls_acc = (preds == labels).float().mean().item()

    iou = pixel_iou(pred_mask, gt_mask.unsqueeze(1), threshold)
    f1 = pixel_f1(pred_mask, gt_mask.unsqueeze(1), threshold)

    return {"cls_acc": cls_acc, "pixel_iou": iou, "pixel_f1": f1}
