"""Evaluation script — runs inference on the test split and prints metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from data import ForgeryDataset, get_val_transforms
from models import ForgeryDetector
from utils import compute_metrics


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate(model, loader, device, cfg) -> dict[str, float]:
    model.eval()
    all_metrics: dict[str, list] = {"cls_acc": [], "pixel_iou": [], "pixel_f1": []}
    for images, masks, labels in tqdm(loader, desc="eval"):
        images = images.to(device)
        masks = masks.to(device).float()
        labels = labels.to(device)
        logits, pred_mask = model(images)
        batch_metrics = compute_metrics(
            logits, pred_mask, labels, masks, cfg["eval"]["threshold"]
        )
        for k, v in batch_metrics.items():
            all_metrics[k].append(v)

    return {k: float(sum(v) / len(v)) for k, v in all_metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate forgery detector on the test split")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

    model = ForgeryDetector(
        backbone=cfg["model"]["backbone"],
        pretrained=False,
        num_classes=cfg["model"]["num_classes"],
        decoder_channels=cfg["model"]["decoder_channels"],
    )
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    test_ds = ForgeryDataset(
        cfg["data"]["root"],
        split="test",
        transform=get_val_transforms(cfg["data"]["image_size"]),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    metrics = evaluate(model, test_loader, device, cfg)
    print("\n── Test Results ──────────────────────────────────")
    for k, v in metrics.items():
        print(f"  {k:<15} {v:.4f}")


if __name__ == "__main__":
    main()
