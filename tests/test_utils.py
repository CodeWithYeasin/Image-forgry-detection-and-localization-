"""Tests for utility metrics."""

import sys
from pathlib import Path

import torch
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.metrics import pixel_iou, pixel_f1, compute_metrics


class TestPixelIou:
    def test_perfect_prediction(self):
        mask = torch.ones(2, 1, 8, 8) * 10.0   # high logit → sigmoid ≈ 1
        gt = torch.ones(2, 1, 8, 8)
        assert pixel_iou(mask, gt) == pytest.approx(1.0, abs=1e-3)

    def test_no_overlap(self):
        pred = torch.ones(1, 1, 4, 4) * -10.0  # sigmoid ≈ 0
        gt = torch.ones(1, 1, 4, 4)
        iou = pixel_iou(pred, gt)
        assert iou < 0.01

    def test_shape_flexibility(self):
        pred = torch.zeros(3, 1, 16, 16)
        gt = torch.zeros(3, 1, 16, 16)
        assert pixel_iou(pred, gt) > 0.0   # numerically stable with epsilon


class TestPixelF1:
    def test_perfect_prediction(self):
        mask = torch.ones(2, 1, 8, 8) * 10.0
        gt = torch.ones(2, 1, 8, 8)
        assert pixel_f1(mask, gt) == pytest.approx(1.0, abs=1e-3)

    def test_returns_float(self):
        pred = torch.randn(4, 1, 8, 8)
        gt = torch.randint(0, 2, (4, 1, 8, 8)).float()
        result = pixel_f1(pred, gt)
        assert isinstance(result, float)


class TestComputeMetrics:
    def test_keys_present(self):
        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        labels = torch.tensor([1, 0])
        pred_mask = torch.ones(2, 1, 4, 4) * 5.0
        gt_mask = torch.ones(2, 4, 4)
        metrics = compute_metrics(logits, pred_mask, labels, gt_mask)
        assert set(metrics.keys()) == {"cls_acc", "pixel_iou", "pixel_f1"}

    def test_cls_accuracy_perfect(self):
        logits = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
        labels = torch.tensor([1, 0])
        pred_mask = torch.zeros(2, 1, 4, 4)
        gt_mask = torch.zeros(2, 4, 4)
        metrics = compute_metrics(logits, pred_mask, labels, gt_mask)
        assert metrics["cls_acc"] == pytest.approx(1.0)
