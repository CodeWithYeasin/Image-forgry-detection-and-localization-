"""Inference script — predict forgery mask for one or more images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from data import get_val_transforms
from models import ForgeryDetector
from utils import overlay_mask


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def predict_single(
    model: torch.nn.Module,
    img_path: Path,
    transform,
    device: torch.device,
    threshold: float,
    output_dir: Path,
) -> None:
    image_pil = Image.open(img_path).convert("RGB")
    original_np = np.array(image_pil)
    image_tensor = transform(image=original_np)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits, pred_mask = model(image_tensor)

    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    label = int(probs.argmax())
    confidence = float(probs.max())

    mask_np = torch.sigmoid(pred_mask)[0, 0].cpu().numpy()
    mask_bin = (mask_np > threshold).astype(np.uint8)

    # Resize mask back to original dimensions
    mask_resized = np.array(
        Image.fromarray((mask_np * 255).astype(np.uint8)).resize(
            (original_np.shape[1], original_np.shape[0]), Image.BILINEAR
        )
    )
    mask_bin_resized = (mask_resized > int(threshold * 255)).astype(np.uint8)

    vis = overlay_mask(original_np, mask_bin_resized)
    out_path = output_dir / (img_path.stem + "_prediction.png")
    Image.fromarray(vis).save(out_path)

    cls_name = "FORGED" if label == 1 else "AUTHENTIC"
    print(f"{img_path.name}  →  {cls_name} ({confidence:.2%})   saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forgery detection inference on images")
    parser.add_argument("images", nargs="+", help="One or more image file paths")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth)")
    parser.add_argument("--output_dir", default="predictions", help="Directory for output images")
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
    model.to(device).eval()

    transform = get_val_transforms(cfg["data"]["image_size"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in args.images:
        predict_single(
            model, Path(img_path), transform, device, cfg["eval"]["threshold"], output_dir
        )


if __name__ == "__main__":
    main()
