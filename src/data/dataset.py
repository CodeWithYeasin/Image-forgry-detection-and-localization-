"""PyTorch Dataset for image forgery detection and localization."""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ForgeryDataset(Dataset):
    """Dataset that loads image–mask pairs for forgery detection.

    Expected directory layout::

        root/
          authentic/
            image1.jpg
            image2.png
            ...
          forged/
            image1.jpg   # manipulated image
            image1_mask.png  # binary ground-truth mask (white = forged region)
            ...

    Args:
        root: Path to the dataset root directory.
        split: One of ``'train'``, ``'val'``, or ``'test'``.
        transform: Albumentations transform applied to the image and mask.
        return_mask: If ``True``, also return the binary forgery mask.
    """

    AUTHENTIC = 0
    FORGED = 1

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
        return_mask: bool = True,
    ) -> None:
        self.root = Path(root) / split
        self.transform = transform
        self.return_mask = return_mask

        authentic_dir = self.root / "authentic"
        forged_dir = self.root / "forged"

        self.samples: list[Tuple[Path, Optional[Path], int]] = []

        if authentic_dir.is_dir():
            for img_path in sorted(authentic_dir.glob("*")):
                if self._is_image(img_path):
                    self.samples.append((img_path, None, self.AUTHENTIC))

        if forged_dir.is_dir():
            for img_path in sorted(forged_dir.glob("*")):
                if self._is_image(img_path) and "_mask" not in img_path.stem:
                    mask_path = img_path.with_name(img_path.stem + "_mask.png")
                    mask_path = mask_path if mask_path.exists() else None
                    self.samples.append((img_path, mask_path, self.FORGED))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path, label = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.return_mask and mask_path is not None:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask, label

    # ------------------------------------------------------------------
    @staticmethod
    def _is_image(path: Path) -> bool:
        return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
