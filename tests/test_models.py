"""Smoke tests for the ForgeryDetector model."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestForgeryDetectorSmoke:
    """Lightweight tests that do NOT download pretrained weights."""

    @pytest.fixture
    def model(self):
        from models import ForgeryDetector
        return ForgeryDetector(backbone="resnet18", pretrained=False, num_classes=2)

    def test_output_shapes(self, model):
        x = torch.randn(2, 3, 256, 256)
        logits, mask = model(x)
        assert logits.shape == (2, 2)
        assert mask.shape == (2, 1, 256, 256)

    def test_logits_finite(self, model):
        x = torch.randn(1, 3, 128, 128)
        logits, mask = model(x)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(mask).all()

    def test_different_input_sizes(self, model):
        for size in [128, 224, 320]:
            x = torch.randn(1, 3, size, size)
            logits, mask = model(x)
            assert mask.shape[-2:] == (size, size)
