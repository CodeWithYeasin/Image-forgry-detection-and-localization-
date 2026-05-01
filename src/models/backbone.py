"""Encoder backbone wrapper built on top of the *timm* library."""

from __future__ import annotations

from typing import List

import timm
import torch
import torch.nn as nn


class EncoderBackbone(nn.Module):
    """Thin wrapper around a *timm* encoder that returns multi-scale features.

    Args:
        name: Any model name accepted by ``timm.create_model``.
        pretrained: Whether to load ImageNet pre-trained weights.
        out_indices: Feature-map stages to return (0-indexed from stem).
    """

    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = True,
        out_indices: tuple[int, ...] = (1, 2, 3, 4),
    ) -> None:
        super().__init__()
        self.encoder = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        self.feature_info = self.encoder.feature_info

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.encoder(x)

    @property
    def out_channels(self) -> List[int]:
        return [fi["num_chs"] for fi in self.feature_info]
