#!/usr/bin/env python3
"""
Train a ResNet18 regression model for road following.

Dataset layout
--------------
  <dataset_dir>/
    labels.csv        # columns: filename,angle  (angle in [-1.0, 1.0])
    img_001.jpg
    img_002.jpg
    ...

The CSV angle is the steering target: negative = left, positive = right.

Usage
-----
  python tools/train_road_follower.py --dataset /path/to/data
  python tools/train_road_follower.py --dataset /path/to/data --epochs 30 --lr 1e-4
  python tools/train_road_follower.py --dataset /path/to/data --output custom_name.pth
"""

import argparse
import csv
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

ASSETS_MODELS = Path(__file__).resolve().parent.parent / "assets" / "models"


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-4
    batch_size: int = 32
    val_split: float = 0.15
    seed: int = 42
    output: Path = field(default_factory=lambda: ASSETS_MODELS / "road_follower.pth")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RoadDataset:
    def __init__(self, rows, transform):
        self._rows = rows
        self._transform = transform

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        import cv2

        img_path, angle = self._rows[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self._transform(img)
        import torch

        return tensor, torch.tensor([float(angle)], dtype=torch.float32)


def _load_rows(dataset_dir: Path):
    csv_path = dataset_dir / "labels.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"labels.csv not found in {dataset_dir}.\n"
            "Expected format: filename,angle  (one row per image)"
        )
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            img_path = dataset_dir / r["filename"]
            angle = float(r["angle"])
            if not -1.0 <= angle <= 1.0:
                raise ValueError(f"Angle {angle} out of [-1, 1] for {r['filename']}")
            rows.append((img_path, angle))
    if not rows:
        raise ValueError("labels.csv is empty")
    return rows


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _build_model():
    import torch
    from torchvision.models import resnet18

    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model


def _build_transform():
    import torchvision.transforms as T

    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def train(dataset_dir: Path, cfg: TrainConfig):
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    rows = _load_rows(dataset_dir)
    random.shuffle(rows)

    n_val = max(1, int(len(rows) * cfg.val_split))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"Dataset: {len(train_rows)} train / {n_val} val  ({len(rows)} total)")

    transform = _build_transform()
    train_loader = DataLoader(
        RoadDataset(train_rows, transform),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        RoadDataset(val_rows, transform),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = _build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.5
    )
    criterion = torch.nn.MSELoss()
    best_val_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, angles in train_loader:
            imgs, angles = imgs.to(device), angles.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = criterion(pred, angles)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_rows)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, angles in val_loader:
                imgs, angles = imgs.to(device), angles.to(device)
                pred = model(imgs)
                val_loss += criterion(pred, angles).item() * len(imgs)
        val_loss /= n_val
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{cfg.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(cfg.output))
            print(f"  -> saved best model  (val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Model saved to: {cfg.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet18 road-follower regression model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", required=True, type=Path, help="Directory containing labels.csv and images"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ASSETS_MODELS / "road_follower.pth",
        help="Output .pth file path",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--val-split", type=float, default=0.15, help="Fraction of data held out for validation"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.dataset.is_dir():
        parser.error(f"Dataset directory not found: {args.dataset}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        batch_size=args.batch_size,
        seed=args.seed,
        output=args.output,
    )
    train(dataset_dir=args.dataset, cfg=cfg)


if __name__ == "__main__":
    main()
