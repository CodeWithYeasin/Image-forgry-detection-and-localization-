"""Test suite for the ForgeryDataset."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import ForgeryDataset


class TestForgeryDataset:
    def test_empty_root_returns_zero_samples(self, tmp_path):
        ds = ForgeryDataset(tmp_path, split="train")
        assert len(ds) == 0

    def test_authentic_samples_loaded(self, tmp_path):
        auth_dir = tmp_path / "train" / "authentic"
        auth_dir.mkdir(parents=True)
        # Create minimal valid PNG (1×1 white pixel)
        from PIL import Image
        img = Image.fromarray(np.ones((4, 4, 3), dtype=np.uint8) * 200)
        img.save(auth_dir / "test.png")

        ds = ForgeryDataset(tmp_path, split="train")
        assert len(ds) == 1
        _, _, label = ds[0]
        assert label == ForgeryDataset.AUTHENTIC

    def test_forged_sample_label(self, tmp_path):
        forged_dir = tmp_path / "train" / "forged"
        forged_dir.mkdir(parents=True)
        from PIL import Image
        img = Image.fromarray(np.ones((4, 4, 3), dtype=np.uint8) * 128)
        img.save(forged_dir / "test.png")

        ds = ForgeryDataset(tmp_path, split="train")
        assert len(ds) == 1
        _, _, label = ds[0]
        assert label == ForgeryDataset.FORGED

    def test_mask_loaded_when_present(self, tmp_path):
        forged_dir = tmp_path / "train" / "forged"
        forged_dir.mkdir(parents=True)
        from PIL import Image
        img = Image.fromarray(np.ones((4, 4, 3), dtype=np.uint8) * 128)
        img.save(forged_dir / "test.png")
        mask = Image.fromarray(np.ones((4, 4), dtype=np.uint8) * 255)
        mask.save(forged_dir / "test_mask.png")

        ds = ForgeryDataset(tmp_path, split="train", return_mask=True)
        _, loaded_mask, _ = ds[0]
        assert loaded_mask.sum() > 0

    def test_is_image_filters_non_images(self):
        assert not ForgeryDataset._is_image(Path("file.txt"))
        assert ForgeryDataset._is_image(Path("file.jpg"))
        assert ForgeryDataset._is_image(Path("FILE.PNG"))
