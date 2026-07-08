"""
Train a MobileNetV2-based Braille cell classifier.

Architecture:
  MobileNetV2 pretrained backbone (ImageNet)
  → Global average pool
  → Linear(1280, 6)   — 6 independent dot predictions
  → Sigmoid           (applied at inference; BCEWithLogitsLoss during train)

Labels: 6-bit string, bit[i] = 1 if dot (i+1) is raised.

Output: /tmp/braille-crops/cell_classifier.pt  (state_dict)
        /tmp/braille-crops/training_log.csv
"""

import csv
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

CROP_DIR   = Path('/tmp/braille-crops')
MODEL_OUT  = CROP_DIR / 'cell_classifier.pt'
LOG_OUT    = CROP_DIR / 'training_log.csv'
CROP_SIZE  = 64
BATCH      = 128
EPOCHS     = 30
LR         = 1e-3
LR_PATIENCE = 5     # reduce LR after N epochs without val improvement
EARLY_STOP  = 10    # stop after N epochs without val improvement

device = (
    torch.device('mps')  if torch.backends.mps.is_available() else
    torch.device('cuda') if torch.cuda.is_available() else
    torch.device('cpu')
)


# ── Dataset ─────────────────────────────────────────────────────────────────

class CellDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img  = Image.open(row['path']).convert('RGB')
        img  = self.transform(img)
        bits = torch.tensor([float(b) for b in row['bits6']], dtype=torch.float32)
        return img, bits


def load_manifest(split):
    rows = []
    with open(CROP_DIR / 'manifest.csv') as f:
        for r in csv.DictReader(f):
            if r['split'] == split:
                rows.append(r)
    return rows


def hflip_bits6(bits6):
    """Swap left column (dots 1,2,3) with right column (dots 4,5,6)."""
    return bits6[3:6] + bits6[0:3]


# ── Transforms ──────────────────────────────────────────────────────────────

# MobileNetV2 ImageNet normalisation
NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])

train_tf = transforms.Compose([
    transforms.Resize((CROP_SIZE, CROP_SIZE)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    NORM,
])

val_tf = transforms.Compose([
    transforms.Resize((CROP_SIZE, CROP_SIZE)),
    transforms.ToTensor(),
    NORM,
])


def augment_with_hflip(rows):
    """Double the dataset by adding horizontally-flipped crops with corrected labels."""
    extra = []
    for r in rows:
        flipped = {
            'split':     r['split'],
            'path':      r['path'],       # PIL will flip on-the-fly
            'bits6':     hflip_bits6(r['bits6']),
            'label_int': r['label_int'],
            '_hflip':    True,
        }
        extra.append(flipped)
    return rows + extra


class CellDatasetWithFlip(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform
        self.hflip_tf = transforms.RandomHorizontalFlip(p=1.0)  # always flip when flagged

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(row['path']).convert('RGB')
        if row.get('_hflip'):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img  = self.transform(img)
        bits = torch.tensor([float(b) for b in row['bits6']], dtype=torch.float32)
        return img, bits


# ── Model ────────────────────────────────────────────────────────────────────

def build_model():
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Replace the classifier: 1280 features → 6 dot logits
    m.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(m.last_channel, 6),
    )
    return m


# ── Metrics ──────────────────────────────────────────────────────────────────

def evaluate(model, loader):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    dot_correct = 0
    dot_total   = 0
    cell_correct = 0
    cell_total   = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            total_loss += loss_fn(logits, labels).item() * len(imgs)
            preds = (logits.sigmoid() > 0.5).float()
            dot_correct  += (preds == labels).sum().item()
            dot_total    += labels.numel()
            cell_correct += (preds == labels).all(dim=1).sum().item()
            cell_total   += len(imgs)

    return {
        'loss':      total_loss / cell_total,
        'dot_acc':   dot_correct / dot_total,
        'cell_acc':  cell_correct / cell_total,
    }


# ── Training ─────────────────────────────────────────────────────────────────

def main():
    # Load data
    train_rows = augment_with_hflip(load_manifest('train'))
    val_rows   = load_manifest('val')
    test_rows  = load_manifest('test')
    print(f"Device: {device}")
    print(f"Train: {len(train_rows)} crops  Val: {len(val_rows)}  Test: {len(test_rows)}")

    train_ds = CellDatasetWithFlip(train_rows, train_tf)
    val_ds   = CellDataset(val_rows, val_tf)
    test_ds  = CellDataset(test_rows, val_tf)

    # num_workers=0 avoids macOS shared-memory timeout with MPS
    nw = 0 if str(device) == 'mps' else 4
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=nw, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0,  pin_memory=False)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0,  pin_memory=False)

    model   = build_model().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optim   = torch.optim.Adam(model.parameters(), lr=LR)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(
                  optim, mode='min', factor=0.5, patience=LR_PATIENCE)

    best_val_loss = float('inf')
    no_improve    = 0
    log_rows      = []

    print(f"\n{'Epoch':>5}  {'TrainLoss':>9}  {'ValLoss':>8}  {'ValDotAcc':>9}  {'ValCellAcc':>10}  {'LR':>8}")
    print('-' * 65)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)
            optim.zero_grad()
            loss = loss_fn(model(imgs), labels)
            loss.backward()
            optim.step()
            total_loss += loss.item() * len(imgs)
        train_loss = total_loss / len(train_rows)

        val_m  = evaluate(model, val_dl)
        cur_lr = optim.param_groups[0]['lr']
        sched.step(val_m['loss'])

        improved = val_m['loss'] < best_val_loss
        if improved:
            best_val_loss = val_m['loss']
            no_improve    = 0
            torch.save(model.state_dict(), MODEL_OUT)
            flag = ' ✓'
        else:
            no_improve += 1
            flag = ''

        print(f"{epoch:5d}  {train_loss:9.4f}  {val_m['loss']:8.4f}  "
              f"{val_m['dot_acc']:9.4f}  {val_m['cell_acc']:10.4f}  {cur_lr:.2e}{flag}")

        log_rows.append({
            'epoch': epoch, 'train_loss': train_loss,
            'val_loss': val_m['loss'], 'val_dot_acc': val_m['dot_acc'],
            'val_cell_acc': val_m['cell_acc'], 'lr': cur_lr,
        })

        if no_improve >= EARLY_STOP:
            print(f"\nEarly stop at epoch {epoch} (no improvement for {EARLY_STOP} epochs)")
            break

    # Final test evaluation
    model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    test_m = evaluate(model, test_dl)
    print(f"\nTest set — dot_acc: {test_m['dot_acc']:.4f}  cell_acc: {test_m['cell_acc']:.4f}")

    # Save log
    with open(LOG_OUT, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        w.writeheader()
        w.writerows(log_rows)
    print(f"Model saved to {MODEL_OUT}")
    print(f"Log   saved to {LOG_OUT}")


if __name__ == '__main__':
    main()
