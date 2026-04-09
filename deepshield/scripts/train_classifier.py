"""
scripts/train_classifier.py
─────────────────────────────────────────────────────────
DeepShield KYC  –  Fine-tune EfficientNet-B4 on FaceForensics++

Usage:
  python scripts/train_classifier.py \
    --data  /path/to/faceforensics \
    --out   models/deepfake_effnetb4_ff++.pth \
    --epochs 10 \
    --batch  16

Dataset structure expected:
  /data/
    real/   *.jpg (real faces)
    fake/   *.jpg (manipulated faces)

This script works for hackathon conditions — fast training
on a subset of FF++ data gets you ~88%+ AUC in 2-4 hours
on a single GPU.
─────────────────────────────────────────────────────────
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, random_split
    import torchvision.transforms as T
    import timm
    from sklearn.metrics import roc_auc_score
    TORCH_OK = True
except ImportError:
    print("ERROR: PyTorch not installed. Run: pip install torch torchvision timm scikit-learn")
    TORCH_OK = False


# ── Dataset ──────────────────────────────────────────────────────────────────

class FaceDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, max_per_class: int = 5000):
        self.transform = transform
        self.samples   = []

        data_path = Path(data_dir)

        real_dir = data_path / "real"
        fake_dir = data_path / "fake"

        exts = {".jpg", ".jpeg", ".png"}

        for p in list(real_dir.glob("**/*"))[:max_per_class]:
            if p.suffix.lower() in exts:
                self.samples.append((str(p), 0))   # 0 = real

        for p in list(fake_dir.glob("**/*"))[:max_per_class]:
            if p.suffix.lower() in exts:
                self.samples.append((str(p), 1))   # 1 = fake

        np.random.shuffle(self.samples)
        print(f"Dataset: {sum(1 for _,l in self.samples if l==0)} real, "
              f"{sum(1 for _,l in self.samples if l==1)} fake")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


# ── Transforms ───────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Training ─────────────────────────────────────────────────────────────────

def train(args):
    if not TORCH_OK:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Model
    print("Loading EfficientNet-B4...")
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=1)
    model.to(device)

    # ── Dataset
    full_ds = FaceDataset(args.data, TRAIN_TRANSFORM, max_per_class=args.max_per_class)
    val_size  = max(1, int(len(full_ds) * 0.15))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    val_ds.dataset.transform = VAL_TRANSFORM   # override transform for val

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=4, pin_memory=True)

    # ── Optimizer & loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        avg_loss = train_loss / len(train_loader)

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                logits = model(imgs).squeeze(1)
                probs  = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(labels.numpy())

        auc = roc_auc_score(all_labels, all_preds)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:02d}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"AUC: {auc:.4f} | Time: {elapsed:.1f}s")

        # Save best
        if auc > best_auc:
            best_auc = auc
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(out_path))
            print(f"  ✓ Saved best checkpoint (AUC={best_auc:.4f}) → {out_path}")

    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    print(f"Checkpoint saved at: {args.out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DeepShield deepfake classifier")
    parser.add_argument("--data",          type=str, required=True, help="Path to data dir (real/ fake/)")
    parser.add_argument("--out",           type=str, default="models/deepfake_effnetb4_ff++.pth")
    parser.add_argument("--epochs",        type=int, default=10)
    parser.add_argument("--batch",         type=int, default=16)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--max-per-class", type=int, default=5000, dest="max_per_class")
    args = parser.parse_args()

    train(args)
