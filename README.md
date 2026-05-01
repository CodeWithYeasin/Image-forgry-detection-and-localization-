# Image Forgery Detection and Localization

A deep-learning pipeline for detecting and spatially localizing image forgeries (splicing, copy-move, inpainting, etc.).  
The model produces both an **image-level classification** (authentic vs. forged) and a **pixel-level forgery mask**.

---

## ✨ Features

- Encoder–decoder architecture with a *timm* backbone (default: ResNet-50)
- Dual-branch output: classification head + U-Net-style segmentation head
- Albumentations augmentation pipeline
- Mixed-precision training (AMP), early stopping, TensorBoard logging
- Config-driven via YAML — no hard-coded hyper-parameters

---

## 🗂️ Project Structure

```
├── configs/
│   └── default.yaml          # All hyper-parameters in one place
├── data/
│   ├── raw/                  # Original dataset images (git-ignored)
│   ├── processed/            # Pre-processed images  (git-ignored)
│   └── augmented/            # Augmented copies      (git-ignored)
├── docs/                     # Additional documentation
├── notebooks/                # Jupyter notebooks for exploration & analysis
├── src/
│   ├── data/
│   │   ├── dataset.py        # ForgeryDataset (PyTorch Dataset)
│   │   └── transforms.py     # Albumentations pipelines
│   ├── models/
│   │   ├── backbone.py       # timm encoder wrapper
│   │   └── forgery_detector.py  # Full encoder-decoder model
│   ├── utils/
│   │   ├── metrics.py        # IoU, F1, accuracy helpers
│   │   └── visualization.py  # Mask overlay & grid saving
│   ├── train.py              # Training entry-point
│   ├── evaluate.py           # Test-set evaluation
│   └── predict.py            # Single-image inference
├── tests/                    # Pytest unit tests
├── weights/                  # Saved model weights  (git-ignored)
├── .gitignore
├── requirements.txt
└── setup.py
```

---

## 🚀 Quick Start

### 1 — Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2 — Prepare the dataset

Place images under `data/raw/` following the layout expected by `ForgeryDataset`:

```
data/raw/
  train/
    authentic/  ← real images
    forged/     ← manipulated images  +  *_mask.png ground-truth masks
  val/  (same structure)
  test/ (same structure)
```

### 3 — Train

```bash
python src/train.py --config configs/default.yaml
```

### 4 — Evaluate on the test split

```bash
python src/evaluate.py --config configs/default.yaml --weights weights/best_model.pth
```

### 5 — Predict on new images

```bash
python src/predict.py image1.jpg image2.png \
    --config configs/default.yaml \
    --weights weights/best_model.pth \
    --output_dir predictions/
```

---

## ⚙️ Configuration

All settings live in `configs/default.yaml`.  
Key knobs:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `data` | `image_size` | 256 | Spatial resolution fed to the network |
| `model` | `backbone` | `resnet50` | Any *timm*-compatible encoder |
| `train` | `epochs` | 50 | Maximum training epochs |
| `train` | `learning_rate` | 1e-4 | Initial AdamW learning rate |
| `train` | `mixed_precision` | true | Enable AMP for faster training |
| `eval` | `threshold` | 0.5 | Mask binarization threshold |

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 📄 License

[MIT](LICENSE) — © 2026 Md Yeasin Arafat
