"""Albumentations-based augmentation pipelines."""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size: int = 256) -> A.Compose:
    """Return the augmentation pipeline used during training."""
    return A.Compose(
        [
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussNoise(p=0.3),
            A.GaussianBlur(p=0.2),
            A.CLAHE(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size: int = 256) -> A.Compose:
    """Return the deterministic pipeline used during validation and testing."""
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
