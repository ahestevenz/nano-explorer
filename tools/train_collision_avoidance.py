#!/usr/bin/env python3
"""
Train a ResNet18 binary classifier for collision avoidance.

Dataset layout (produced by collect_collision_data.py)
------------------------------------------------------
  <dataset_dir>/
    free/
      img_000001.jpg
      ...
    blocked/
      img_000001.jpg
      ...

Label convention: free=0, blocked=1.
The saved state dict is compatible with lib/motion/collision.py which reads
  torch.softmax(out, dim=1)[0][1]  as the "blocked" probability.

Usage
-----
  python tools/train_collision_avoidance.py --dataset datasets/collision_001
  python tools/train_collision_avoidance.py --dataset datasets/collision_001 --epochs 15 --lr 1e-4
  python tools/train_collision_avoidance.py --dataset datasets/collision_001 --output custom.pth

Required packages
------------------
  Linux (Jetson, JetPack 4.6.1 / CUDA 10.2):
    - torch==1.10.0        (install manually — see doc/jetbot-setup.md)
    - torchvision==0.11.0  (build from source — see doc/jetbot-setup.md)
    - OpenCV 4.1.1 (bundled with JetPack — do not reinstall via pip)

  macOS (CPU or Apple Silicon MPS):
    - pip install torch torchvision
    - pip install opencv-python
"""

import argparse
import os
import random
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("OPENBLAS_CORETYPE", "ARMV8")

ASSETS_MODELS = Path(__file__).resolve().parent.parent / "assets" / "models"

# free=0, blocked=1 — must match collision.py inference (softmax index 1 = blocked)
_LABELS = {"free": 0, "blocked": 1}


@dataclass
class TrainConfig:
    epochs: int = 15
    lr: float = 1e-4
    batch_size: int = 32
    val_split: float = 0.15
    seed: int = 42
    output: Path = field(default_factory=lambda: ASSETS_MODELS / "collision_avoidance.pth")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CollisionDataset:
    def __init__(self, rows, transform):
        self._rows = rows
        self._transform = transform

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        import cv2
        import torch

        img_path, label = self._rows[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self._transform(img)
        return tensor, torch.tensor(label, dtype=torch.long)


def _load_rows(dataset_dir: Path):
    rows = []
    for label_name, label_idx in _LABELS.items():
        folder = dataset_dir / label_name
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Expected subdirectory not found: {folder}\n"
                "Run collect_collision_data.py first."
            )
        images = sorted(folder.glob("*.jpg"))
        if not images:
            raise ValueError(f"No images found in {folder}")
        for img_path in images:
            rows.append((img_path, label_idx))

    free_count = sum(1 for _, lbl in rows if lbl == 0)
    blocked_count = sum(1 for _, lbl in rows if lbl == 1)
    print(f"Dataset: {free_count} free  {blocked_count} blocked  ({len(rows)} total)")
    return rows


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _build_model():
    import torch
    from torchvision.models import resnet18

    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model


def _build_transform():
    import torchvision.transforms as T

    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomHorizontalFlip(),
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
    print(f"Split:   {len(train_rows)} train / {n_val} val")

    transform = _build_transform()
    train_loader = DataLoader(
        CollisionDataset(train_rows, transform),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        CollisionDataset(val_rows, transform),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device:  {device}")

    model = _build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.5
    )
    criterion = torch.nn.CrossEntropyLoss()
    best_val_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss, train_correct = 0.0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(imgs)
            train_correct += (out.argmax(dim=1) == labels).sum().item()
        train_loss /= len(train_rows)
        train_acc = train_correct / len(train_rows)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_loss += criterion(out, labels).item() * len(imgs)
                val_correct += (out.argmax(dim=1) == labels).sum().item()
        val_loss /= n_val
        val_acc = val_correct / n_val
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(cfg.output))
            print(f"  -> saved best model  (val_acc={best_val_acc:.3f})")

    print(f"\nTraining complete.  Best val_acc={best_val_acc:.3f}")
    print(f"Model saved to: {cfg.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet18 collision-avoidance binary classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Directory containing free/ and blocked/ subdirectories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ASSETS_MODELS / "collision_avoidance.pth",
        help="Output .pth file path",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        dest="val_split",
        help="Fraction of data held out for validation",
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
