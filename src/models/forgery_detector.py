"""Dual-branch forgery detector: classification + segmentation (localization)."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import EncoderBackbone


class _DecoderBlock(nn.Module):
    """Single upsampling block for the segmentation head."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ForgeryDetector(nn.Module):
    """Encoder–decoder network for simultaneous forgery classification and mask prediction.

    The model produces:
    - ``logits`` – shape ``(B, num_classes)`` for image-level classification.
    - ``mask``   – shape ``(B, 1, H, W)`` with pixel-wise forgery probability.

    Args:
        backbone: Name of the *timm* encoder backbone.
        pretrained: Load ImageNet weights for the backbone.
        num_classes: Number of output classes for the classification head.
        decoder_channels: Number of feature channels at each decoder stage.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 2,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32),
    ) -> None:
        super().__init__()
        self.encoder = EncoderBackbone(backbone, pretrained=pretrained)
        enc_channels = self.encoder.out_channels  # e.g. [256, 512, 1024, 2048]

        # ── Segmentation decoder (U-Net style) ────────────────────────────
        decoder_blocks = []
        in_ch = enc_channels[-1]
        for i, out_ch in enumerate(decoder_channels):
            skip_ch = enc_channels[-(i + 2)] if i + 2 <= len(enc_channels) else 0
            decoder_blocks.append(_DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        self.decoder = nn.ModuleList(decoder_blocks)
        self.seg_head = nn.Conv2d(decoder_channels[-1], 1, kernel_size=1)

        # ── Classification head ───────────────────────────────────────────
        self.cls_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)          # list of multi-scale feature maps

        # Classification
        logits = self.cls_head(self.cls_pool(features[-1]))

        # Segmentation
        d = features[-1]
        skips = list(reversed(features[:-1]))
        for i, block in enumerate(self.decoder):
            skip = skips[i] if i < len(skips) else torch.zeros_like(d)
            d = block(d, skip)
        mask = F.interpolate(
            self.seg_head(d), size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        return logits, mask
