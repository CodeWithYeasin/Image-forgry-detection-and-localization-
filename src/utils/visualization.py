"""Visualization helpers for forgery masks and prediction overlays."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a binary forgery mask onto the original image.

    Args:
        image: RGB image, shape ``(H, W, 3)``, dtype ``uint8``.
        mask: Binary mask, shape ``(H, W)``, values in ``{0, 1}``.
        color: RGB color used to highlight the forged region.
        alpha: Transparency of the overlay (0 = fully transparent, 1 = opaque).

    Returns:
        Blended image as a ``uint8`` NumPy array.
    """
    overlay = image.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask.astype(bool)] = (
        overlay[mask.astype(bool)] * (1 - alpha) + color_arr * alpha
    )
    return overlay.clip(0, 255).astype(np.uint8)


def save_grid(
    images: torch.Tensor,
    save_path: str | Path,
    nrow: int = 8,
    normalize: bool = True,
    titles: Optional[Sequence[str]] = None,
) -> None:
    """Save a batch of tensors as an image grid.

    Args:
        images: Float tensor of shape ``(N, C, H, W)``.
        save_path: Destination file path (PNG recommended).
        nrow: Number of images per grid row.
        normalize: Rescale pixel values to ``[0, 1]``.
        titles: Unused placeholder kept for API consistency.
    """
    grid = vutils.make_grid(images, nrow=nrow, normalize=normalize, scale_each=True)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    grid_np = (grid_np * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(grid_np).save(save_path)
