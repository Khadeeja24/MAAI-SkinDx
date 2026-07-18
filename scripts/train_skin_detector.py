# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Skin Detector — MobileNetV3 Training Script
# ══════════════════════════════════════════════════════════════════
# Trains a binary skin/not-skin classifier
# using fine-tuned MobileNetV3-Small pretrained on ImageNet.
#
# Input  : 1000 skin images + 900 not-skin images
# Output : trained model saved to models/skin_detector/
#          skin_detector.pth
# ══════════════════════════════════════════════════════════════════

import os
import sys
import time
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# ─── Reproducibility — fix all random seeds ───────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ─── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKIN_DIR     = os.path.join(PROJECT_ROOT, "data", "raw",
               "skin_detector_training", "skin")
NOT_SKIN_DIR = os.path.join(PROJECT_ROOT, "data", "raw",
               "skin_detector_training", "not_skin", "selected")
TRAIN_DIR    = os.path.join(PROJECT_ROOT, "data", "raw",
               "skin_detector_training", "train")
VAL_DIR      = os.path.join(PROJECT_ROOT, "data", "raw",
               "skin_detector_training", "val")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models", "skin_detector")
MODEL_PATH   = os.path.join(MODEL_DIR, "skin_detector.pth")

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Settings ──────────────────────────────────────────────────────
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS     = 15
LR         = 0.001
VAL_RATIO  = 0.2
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'═' * 55}")
print(f"  MAAI-SkinDx | Skin Detector Training")
print(f"{'═' * 55}")
print(f"  Device      : {DEVICE}")
print(f"  Image size  : {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"  Batch size  : {BATCH_SIZE}")
print(f"  Epochs      : {EPOCHS}")
print(f"  Random seed : {SEED}")
print(f"{'═' * 55}\n")

# ══════════════════════════════════════════════════════════════════
# STEP 1 — Prepare train/val split
# ══════════════════════════════════════════════════════════════════
print("── Step 1: Preparing train/val split ──")

def prepare_split(skin_dir, not_skin_dir, train_dir, val_dir,
                  val_ratio=VAL_RATIO, seed=SEED):
    """
    Split images into train and val folders.
    Skips if folders already exist and are populated.
    """
    # Check if already done
    train_skin_dir = os.path.join(train_dir, "skin")
    val_skin_dir   = os.path.join(val_dir, "skin")

    if (os.path.exists(train_skin_dir) and
        len(os.listdir(train_skin_dir)) > 0):
        train_count = sum(
            len(list(Path(train_dir).rglob("*.jpg"))) +
            len(list(Path(train_dir).rglob("*.jpeg"))) +
            len(list(Path(train_dir).rglob("*.png")))
            for _ in [1]
        )
        val_count = sum(
            len(list(Path(val_dir).rglob("*.jpg"))) +
            len(list(Path(val_dir).rglob("*.jpeg"))) +
            len(list(Path(val_dir).rglob("*.png")))
            for _ in [1]
        )
        print(f"  Train/val split already exists — skipping")
        print(f"  Train : {train_count} images")
        print(f"  Val   : {val_count} images")
        return

    random.seed(seed)

    # Collect skin images
    skin_images = []
    for subfolder in ['dermoscopic', 'clinical']:
        folder = os.path.join(skin_dir, subfolder)
        if os.path.exists(folder):
            imgs = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            skin_images.extend(imgs)

    # Collect not-skin images
    not_skin_images = [
        os.path.join(not_skin_dir, f)
        for f in os.listdir(not_skin_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

    print(f"  Total skin images     : {len(skin_images)}")
    print(f"  Total not-skin images : {len(not_skin_images)}")

    # Split each class
    def split(images, val_ratio):
        random.shuffle(images)
        n_val = int(len(images) * val_ratio)
        return images[n_val:], images[:n_val]

    skin_train,     skin_val     = split(skin_images,     val_ratio)
    not_skin_train, not_skin_val = split(not_skin_images, val_ratio)

    print(f"\n  Train — skin: {len(skin_train)}  "
          f"not_skin: {len(not_skin_train)}")
    print(f"  Val   — skin: {len(skin_val)}  "
          f"not_skin: {len(not_skin_val)}")

    # Copy into folder structure
    for split_name, skin_imgs, not_skin_imgs in [
        ('train', skin_train, not_skin_train),
        ('val',   skin_val,   not_skin_val)
    ]:
        for class_name, imgs in [
            ('skin',     skin_imgs),
            ('not_skin', not_skin_imgs)
        ]:
            out_dir = os.path.join(
                PROJECT_ROOT, "data", "raw",
                "skin_detector_training",
                split_name, class_name
            )
            os.makedirs(out_dir, exist_ok=True)

            for img_path in imgs:
                fname    = os.path.basename(img_path)
                out_path = os.path.join(out_dir, fname)
                if not os.path.exists(out_path):
                    shutil.copy2(img_path, out_path)

    print(f"\n  Train/val split complete")

prepare_split(SKIN_DIR, NOT_SKIN_DIR, TRAIN_DIR, VAL_DIR)

# ══════════════════════════════════════════════════════════════════
# STEP 2 — Data transforms and loaders
# ══════════════════════════════════════════════════════════════════
print("\n── Step 2: Setting up data loaders ──")

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
val_dataset   = datasets.ImageFolder(VAL_DIR,   transform=val_transforms)

print(f"  Classes : {train_dataset.classes}")
print(f"  Train   : {len(train_dataset)} images")
print(f"  Val     : {len(val_dataset)} images")

# Handle class imbalance with weighted sampling
class_counts = [0, 0]
for _, label in train_dataset:
    class_counts[label] += 1

weights      = [1.0 / class_counts[label] for _, label in train_dataset]
sampler      = WeightedRandomSampler(weights, len(weights))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)
val_loader   = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ══════════════════════════════════════════════════════════════════
# STEP 3 — Build MobileNetV3-Small model
# ══════════════════════════════════════════════════════════════════
print("\n── Step 3: Building MobileNetV3-Small model ──")

model = models.mobilenet_v3_small(
    weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final classifier with binary skin/not-skin classifier
model.classifier = nn.Sequential(
    nn.Linear(576, 256),
    nn.Hardswish(),
    nn.Dropout(p=0.3),
    nn.Linear(256, 2)
)

# Unfreeze last 3 feature layers for fine-tuning
for param in model.features[-3:].parameters():
    param.requires_grad = True

model = model.to(DEVICE)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

print(f"  Architecture         : MobileNetV3-Small")
print(f"  Total parameters     : {total_params:,}")
print(f"  Trainable parameters : {trainable_params:,}")
print(f"  Frozen parameters    : {total_params - trainable_params:,}")

# ══════════════════════════════════════════════════════════════════
# STEP 4 — Training loop
# ══════════════════════════════════════════════════════════════════
print("\n── Step 4: Training ──")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.5
)

best_val_acc = 0.0
best_epoch   = 0

for epoch in range(EPOCHS):
    start_time = time.time()

    # ── Training phase ─────────────────────────────────────────
    model.train()
    train_loss    = 0.0
    train_correct = 0
    train_total   = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item()
        preds          = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total   += labels.size(0)

    train_acc = train_correct / train_total

    # ── Validation phase ───────────────────────────────────────
    model.eval()
    val_correct = 0
    val_total   = 0
    all_preds   = []
    all_labels  = []

    with torch.no_grad():
        for images, labels in val_loader:
            images  = images.to(DEVICE)
            labels  = labels.to(DEVICE)
            outputs = model(images)

            preds        = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc = val_correct / val_total
    elapsed = time.time() - start_time

    print(f"  Epoch [{epoch+1:2d}/{EPOCHS}]  "
          f"Loss: {train_loss/len(train_loader):.4f}  "
          f"Train: {train_acc:.4f}  "
          f"Val: {val_acc:.4f}  "
          f"Time: {elapsed:.1f}s")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch + 1
        torch.save({
            'epoch'      : epoch,
            'model_state': model.state_dict(),
            'val_acc'    : val_acc,
            'classes'    : train_dataset.classes,
            'seed'       : SEED,
        }, MODEL_PATH)
        print(f"  ✓ New best model saved (val_acc={val_acc:.4f})")

    scheduler.step()

# ══════════════════════════════════════════════════════════════════
# STEP 5 — Final evaluation
# ══════════════════════════════════════════════════════════════════
print(f"\n── Step 5: Final Evaluation ──")
print(f"  Best validation accuracy : {best_val_acc:.4f} "
      f"(epoch {best_epoch}/{EPOCHS})")

print(f"\n  Classification Report:")
print(classification_report(
    all_labels, all_preds,
    target_names=train_dataset.classes
))

cm = confusion_matrix(all_labels, all_preds)
print(f"  Confusion Matrix:")
print(f"                 Predicted NOT_SKIN    Predicted SKIN")
print(f"  Actual NOT_SKIN     {cm[0][0]}                  {cm[0][1]}")
print(f"  Actual SKIN         {cm[1][0]}                  {cm[1][1]}")

print(f"\n{'═' * 55}")
print(f"  Training Complete")
print(f"  Model saved to : {MODEL_PATH}")
print(f"  Val accuracy   : {best_val_acc:.4f}")
print(f"  Random seed    : {SEED}")
print(f"{'═' * 55}\n")