"""Training entry-point for the image forgery detection model."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from data import ForgeryDataset, get_train_transforms, get_val_transforms
from models import ForgeryDetector
from utils import compute_metrics


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cls_criterion: nn.Module,
    seg_criterion: nn.Module,
    device: torch.device,
    cfg: dict,
    scaler: torch.cuda.amp.GradScaler,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    all_metrics: dict[str, list] = {"cls_acc": [], "pixel_iou": [], "pixel_f1": []}

    for images, masks, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device).float()
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg["train"]["mixed_precision"]):
            logits, pred_mask = model(images)
            cls_loss = cls_criterion(logits, labels)
            seg_loss = seg_criterion(pred_mask.squeeze(1), masks)
            loss = (
                cfg["loss"]["cls_weight"] * cls_loss
                + cfg["loss"]["seg_weight"] * seg_loss
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        batch_metrics = compute_metrics(logits, pred_mask, labels, masks)
        for k, v in batch_metrics.items():
            all_metrics[k].append(v)

    return {
        "loss": total_loss / len(loader),
        **{k: float(np.mean(v)) for k, v in all_metrics.items()},
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cls_criterion: nn.Module,
    seg_criterion: nn.Module,
    device: torch.device,
    cfg: dict,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_metrics: dict[str, list] = {"cls_acc": [], "pixel_iou": [], "pixel_f1": []}

    for images, masks, labels in tqdm(loader, desc="val", leave=False):
        images = images.to(device)
        masks = masks.to(device).float()
        labels = labels.to(device)

        logits, pred_mask = model(images)
        cls_loss = cls_criterion(logits, labels)
        seg_loss = seg_criterion(pred_mask.squeeze(1), masks)
        loss = (
            cfg["loss"]["cls_weight"] * cls_loss
            + cfg["loss"]["seg_weight"] * seg_loss
        )

        total_loss += loss.item()
        batch_metrics = compute_metrics(logits, pred_mask, labels, masks)
        for k, v in batch_metrics.items():
            all_metrics[k].append(v)

    return {
        "loss": total_loss / len(loader),
        **{k: float(np.mean(v)) for k, v in all_metrics.items()},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the forgery detector")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["train"]["seed"])

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets & loaders
    train_ds = ForgeryDataset(
        cfg["data"]["root"], split="train", transform=get_train_transforms(cfg["data"]["image_size"])
    )
    val_ds = ForgeryDataset(
        cfg["data"]["root"], split="val", transform=get_val_transforms(cfg["data"]["image_size"])
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=cfg["data"]["pin_memory"],
    )

    # Model, loss, optimiser
    model = ForgeryDetector(
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        num_classes=cfg["model"]["num_classes"],
        decoder_channels=cfg["model"]["decoder_channels"],
    ).to(device)

    cls_criterion = nn.CrossEntropyLoss()
    seg_criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["mixed_precision"])

    # Logging
    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg["paths"]["logs_dir"])

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_stats = train_one_epoch(
            model, train_loader, optimizer, cls_criterion, seg_criterion, device, cfg, scaler
        )
        val_stats = evaluate(model, val_loader, cls_criterion, seg_criterion, device, cfg)
        scheduler.step()

        for k, v in train_stats.items():
            writer.add_scalar(f"train/{k}", v, epoch)
        for k, v in val_stats.items():
            writer.add_scalar(f"val/{k}", v, epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_stats['loss']:.4f}  acc={train_stats['cls_acc']:.3f}  "
            f"iou={train_stats['pixel_iou']:.3f} | "
            f"val_loss={val_stats['loss']:.4f}  acc={val_stats['cls_acc']:.3f}  "
            f"iou={val_stats['pixel_iou']:.3f}"
        )

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= cfg["train"]["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    torch.save(model.state_dict(), ckpt_dir / "last_model.pth")
    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
