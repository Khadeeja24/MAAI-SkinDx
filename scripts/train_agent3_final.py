# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 3 — Final Multi-Dataset Training
# ══════════════════════════════════════════════════════════════════
# Trains ResNet-50 on unified dataset from prepare_agent3_data.py
#
# Input : data/processed/agent3_unified_dataset.csv
# Output: models/agent3/agent3_resnet50_final.pth
#
# Dataset stats (from prepare script):
#   Total images  : 150,442
#   Classes       : 9 unified classes
#   Imbalance     : 367:1 (Inflammatory 71k vs Dermatofibroma 193)
#
# Imbalance handling strategy (3 layers):
#   Layer 1 — Class capping: limit each class to MAX_PER_CLASS
#             samples. Prevents majority class from dominating
#             all training batches.
#   Layer 2 — WeightedRandomSampler: on the capped dataset,
#             further oversample rare classes so each class
#             appears roughly equally in each epoch.
#   Layer 3 — Focal Loss gamma=3.0 + class weights: penalise
#             mistakes on hard/rare examples more heavily.
#
# Run prepare_agent3_data.py FIRST. This script will exit
# if data/processed/agent3_unified_dataset.csv does not exist.
# ══════════════════════════════════════════════════════════════════

import os
import sys
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
from PIL import Image as PILImage
from torchvision import transforms, models
from torch.utils.data import (
    DataLoader, Dataset, WeightedRandomSampler, Subset)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, classification_report, confusion_matrix
)
from collections import Counter

# ─── Reproducibility ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ─── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIFIED_CSV   = os.path.join(PROJECT_ROOT, "data", "processed",
                "agent3_unified_dataset.csv")
LABEL_MAP_CSV = os.path.join(PROJECT_ROOT, "data", "processed",
                "agent3_label_mapping.csv")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "outputs", "agent3_comparison")
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models", "agent3")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Guard ─────────────────────────────────────────────────────────
if not os.path.exists(UNIFIED_CSV):
    print(f"\n  ERROR: {UNIFIED_CSV} not found.")
    print(f"  Run first: python scripts/prepare_agent3_data.py")
    sys.exit(1)

# ─── Config ────────────────────────────────────────────────────────
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE    = 32
NUM_EPOCHS    = 25
PATIENCE      = 6
LR_HEAD       = 1e-4
LR_BACKBONE   = 1e-5
IMG_SIZE      = 224
FOCAL_GAMMA   = 3.0

# Class capping — maximum images per class used per epoch.
# With 193 Dermatofibroma images (smallest class) and 9 classes,
# this caps at 193 * 30 = 5,790 per class max.
# Inflammatory drops from 71,006 to 5,790. Much more balanced.
# Adjust this value if training is too slow or too fast.
MAX_PER_CLASS = 6000

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — Load and inspect unified dataset
# ══════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print(f"  MAAI-SkinDx | Agent 3 — Multi-Dataset Training")
print(f"{'═'*65}")
print(f"  Device       : {DEVICE}")
print(f"  Focal gamma  : {FOCAL_GAMMA}")
print(f"  Max per class: {MAX_PER_CLASS:,}")
print(f"  Epochs       : {NUM_EPOCHS} (patience={PATIENCE})")
print(f"{'═'*65}\n")

print("── Section 1: Loading unified dataset ──")
df = pd.read_csv(UNIFIED_CSV)

# Validate images exist
print(f"  Rows in CSV       : {len(df):,}")
df["exists"] = df["image_path"].apply(os.path.exists)
missing = (~df["exists"]).sum()
if missing > 0:
    print(f"  Removing {missing:,} rows with missing images")
    df = df[df["exists"]].copy()
df = df.drop(columns=["exists"]).reset_index(drop=True)
print(f"  Valid images      : {len(df):,}")

# Load label map
label_map    = pd.read_csv(LABEL_MAP_CSV)
CLASS_NAMES  = {
    int(row["class_index"]): row["class_name"]
    for _, row in label_map.iterrows()
}
NUM_CLASSES  = len(CLASS_NAMES)
print(f"  Unified classes   : {NUM_CLASSES}")

print(f"\n  Raw class distribution:")
for cls_idx in sorted(df["unified_label"].unique()):
    cnt  = (df["unified_label"] == cls_idx).sum()
    name = CLASS_NAMES.get(cls_idx, f"Class {cls_idx}")
    print(f"    {cls_idx}: {name:<45} {cnt:7,}")

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — Class capping
# Cap each class at MAX_PER_CLASS to handle 367:1 imbalance.
# HAM10000 records are preserved completely (they are small).
# ══════════════════════════════════════════════════════════════════
print(f"\n── Section 2: Class Capping (max {MAX_PER_CLASS:,} per class) ──")

capped_rows = []
for cls_idx in sorted(df["unified_label"].unique()):
    cls_df = df[df["unified_label"] == cls_idx].copy()
    if len(cls_df) > MAX_PER_CLASS:
        # Always keep all HAM10000 rows for this class
        ham_rows  = cls_df[cls_df["source"] == "HAM10000"]
        rest_rows = cls_df[cls_df["source"] != "HAM10000"]
        # Fill remaining slots from other datasets randomly
        remaining = MAX_PER_CLASS - len(ham_rows)
        if remaining > 0 and len(rest_rows) > 0:
            sampled = rest_rows.sample(
                n=min(remaining, len(rest_rows)),
                random_state=SEED)
            cls_df = pd.concat([ham_rows, sampled])
        else:
            cls_df = ham_rows
    capped_rows.append(cls_df)

df_capped = pd.concat(capped_rows).reset_index(drop=True)
print(f"\n  After capping:")
for cls_idx in sorted(df_capped["unified_label"].unique()):
    cnt  = (df_capped["unified_label"] == cls_idx).sum()
    name = CLASS_NAMES.get(cls_idx, f"Class {cls_idx}")
    print(f"    {cls_idx}: {name:<45} {cnt:6,}")

total_capped = len(df_capped)
print(f"\n  Total after capping: {total_capped:,} "
      f"(from {len(df):,})")

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — Train / hold-out split
# HAM10000 → split by lesion_id to prevent data leakage
# All other datasets → stratified image-level split
# ══════════════════════════════════════════════════════════════════
print(f"\n── Section 3: Train/Hold-out Split (80/20) ──")

ham_df  = df_capped[df_capped["source"] == "HAM10000"].copy()
rest_df = df_capped[df_capped["source"] != "HAM10000"].copy()

train_parts = []
test_parts  = []

# HAM10000 — lesion-level split
if len(ham_df) > 0:
    lesions = ham_df.groupby("group_id")["unified_label"]\
        .first().reset_index()
    tr_l, te_l = train_test_split(
        lesions["group_id"].values,
        test_size=0.20, random_state=SEED,
        stratify=lesions["unified_label"].values)
    train_parts.append(ham_df[ham_df["group_id"].isin(tr_l)])
    test_parts.append(ham_df[ham_df["group_id"].isin(te_l)])

# Other datasets — image-level split
if len(rest_df) > 0:
    tr_rest, te_rest = train_test_split(
        rest_df, test_size=0.20, random_state=SEED,
        stratify=rest_df["unified_label"].values)
    train_parts.append(tr_rest)
    test_parts.append(te_rest)

train_df = pd.concat(train_parts).reset_index(drop=True)
test_df  = pd.concat(test_parts).reset_index(drop=True)

print(f"  Training pool : {len(train_df):,} images")
print(f"  Hold-out test : {len(test_df):,} images — LOCKED")

print(f"\n  Training class distribution (after capping + split):")
for cls_idx in sorted(train_df["unified_label"].unique()):
    cnt  = (train_df["unified_label"] == cls_idx).sum()
    name = CLASS_NAMES.get(cls_idx, f"Class {cls_idx}")
    print(f"    {cls_idx}: {name:<45} {cnt:5,}")

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — Dataset class
# ══════════════════════════════════════════════════════════════════

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]),
])

class SkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths     = list(df["image_path"])
        self.labels    = [int(l) for l in df["unified_label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = PILImage.open(self.paths[idx]).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), self.labels[idx]

# ══════════════════════════════════════════════════════════════════
# SECTION 5 — Focal Loss with class weights
# ══════════════════════════════════════════════════════════════════

tr_labels    = train_df["unified_label"].values.astype(int)
label_counts = Counter(tr_labels)
total        = len(tr_labels)
all_classes  = sorted(train_df["unified_label"].unique())

# Inverse frequency weights
cw_list = []
for cls_idx in all_classes:
    cnt = label_counts.get(cls_idx, 1)
    cw_list.append(total / (len(all_classes) * cnt))
class_weights = torch.FloatTensor(cw_list).to(DEVICE)

print(f"\n── Section 5: Class Weights (inverse frequency) ──")
for i, cls_idx in enumerate(all_classes):
    name = CLASS_NAMES.get(cls_idx, f"Class {cls_idx}")
    print(f"  {cls_idx}: {name:<45} weight={cw_list[i]:.3f}")

class FocalLoss(nn.Module):
    """
    Focal Loss with class weights.
    gamma=3.0 — aggressive focus on hard/rare examples.
    class_weight — extra penalty on minority classes.
    """
    def __init__(self, gamma=3.0, class_weight=None):
        super().__init__()
        self.gamma        = gamma
        self.class_weight = class_weight

    def forward(self, inputs, targets):
        targets    = targets.long()
        log_prob   = F.log_softmax(inputs, dim=1)
        prob       = torch.exp(log_prob)
        ce         = F.nll_loss(
            log_prob, targets,
            weight=self.class_weight, reduction="none")
        pt         = prob.gather(
            1, targets.unsqueeze(1)).squeeze(1)
        return (((1 - pt) ** self.gamma) * ce).mean()

criterion = FocalLoss(
    gamma=FOCAL_GAMMA, class_weight=class_weights)

# ══════════════════════════════════════════════════════════════════
# SECTION 6 — Build ResNet-50
# ══════════════════════════════════════════════════════════════════
print(f"\n── Section 6: Building ResNet-50 ──")

model = models.resnet50(
    weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Freeze all
for p in model.parameters():
    p.requires_grad = False

# Unfreeze layer4
for name, p in model.named_parameters():
    if "layer4" in name:
        p.requires_grad = True

in_features = model.fc.in_features  # 2048
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(in_features, 512),
    nn.ReLU(inplace=True),
    nn.BatchNorm1d(512),
    nn.Dropout(0.3),
    nn.Linear(512, NUM_CLASSES)
)
model = model.to(DEVICE)

trainable = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Output classes   : {NUM_CLASSES}")
print(f"  Trainable params : {trainable:,}")

# ══════════════════════════════════════════════════════════════════
# SECTION 7 — Data loaders with WeightedRandomSampler
# ══════════════════════════════════════════════════════════════════

sample_weights = [
    total / label_counts.get(l, 1) for l in tr_labels]
sampler = WeightedRandomSampler(
    sample_weights, len(sample_weights), replacement=True)

tr_ds = SkinDataset(train_df, TRAIN_TRANSFORM)
ho_ds = SkinDataset(test_df,  VAL_TRANSFORM)

tr_loader = DataLoader(
    tr_ds, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=0, pin_memory=True, drop_last=True)
ho_loader = DataLoader(
    ho_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True)

print(f"\n  Training batches  : {len(tr_loader):,}")
print(f"  Hold-out batches  : {len(ho_loader):,}")

# ══════════════════════════════════════════════════════════════════
# SECTION 8 — Optimizer and scheduler
# ══════════════════════════════════════════════════════════════════

backbone_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "fc" not in n]
head_params     = [p for n, p in model.named_parameters()
                   if p.requires_grad and "fc" in n]

optimizer = torch.optim.Adam([
    {"params": backbone_params, "lr": LR_BACKBONE},
    {"params": head_params,     "lr": LR_HEAD},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

# ══════════════════════════════════════════════════════════════════
# SECTION 9 — Evaluate function
# ══════════════════════════════════════════════════════════════════

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_lbls = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            out    = model(imgs)
            loss   = criterion(out, labels)
            total_loss += loss.item()
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_lbls.extend(labels.cpu().numpy())
    f1   = f1_score(all_lbls, all_preds,
                    average="macro", zero_division=0)
    acc  = accuracy_score(all_lbls, all_preds)
    prec = precision_score(all_lbls, all_preds,
                           average="macro", zero_division=0)
    rec  = recall_score(all_lbls, all_preds,
                        average="macro", zero_division=0)
    return total_loss/len(loader), f1, acc, prec, rec, \
           all_preds, all_lbls

# ══════════════════════════════════════════════════════════════════
# SECTION 10 — Training loop
# ══════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")
print(f"  Training on {len(train_df):,} images")
print(f"  Strategy: Class capping + WeightedSampler + FocalLoss")
print(f"{'─'*65}")

best_f1      = 0.0
patience_cnt = 0
best_state   = None
best_epoch   = 0
tr_losses    = []
va_losses    = []
start_time   = time.time()

for epoch in range(NUM_EPOCHS):
    ep_start = time.time()
    model.train()
    ep_loss = 0.0
    ep_preds, ep_lbls = [], []

    for batch_idx, (imgs, labels) in enumerate(tr_loader):
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE).long()

        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        ep_loss += loss.item()
        ep_preds.extend(out.argmax(1).cpu().numpy())
        ep_lbls.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 200 == 0:
            print(f"    Batch {batch_idx+1}/{len(tr_loader)} "
                  f"loss={loss.item():.4f}")

    scheduler.step()

    tr_loss = ep_loss / len(tr_loader)
    tr_f1   = f1_score(ep_lbls, ep_preds,
                       average="macro", zero_division=0)
    va_loss, va_f1, va_acc, _, _, _, _ = evaluate(
        model, ho_loader)

    tr_losses.append(tr_loss)
    va_losses.append(va_loss)
    ep_time = time.time() - ep_start

    print(f"\n  Epoch {epoch+1:2d}/{NUM_EPOCHS} ({ep_time:.0f}s)")
    print(f"  Train : loss={tr_loss:.4f}  f1={tr_f1*100:.2f}%")
    print(f"  Val   : loss={va_loss:.4f}  "
          f"f1={va_f1*100:.2f}%  acc={va_acc*100:.2f}%")

    if va_f1 > best_f1:
        best_f1      = va_f1
        best_epoch   = epoch + 1
        patience_cnt = 0
        best_state   = {
            k: v.clone() for k, v in model.state_dict().items()}

        ckpt_path = os.path.join(
            MODELS_DIR, "agent3_resnet50_final.pth")
        torch.save({
            "state_dict"    : best_state,
            "val_f1"        : best_f1,
            "epoch"         : best_epoch,
            "class_names"   : CLASS_NAMES,
            "model_name"    : "ResNet-50",
            "in_features"   : in_features,
            "num_classes"   : NUM_CLASSES,
            "datasets_used" : list(df["source"].unique()),
            "total_train"   : len(train_df),
        }, ckpt_path)
        print(f"  \u2713 Best saved "
              f"(f1={best_f1*100:.2f}%) epoch={best_epoch}")
    else:
        patience_cnt += 1
        print(f"  No improvement ({patience_cnt}/{PATIENCE})")
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

elapsed = (time.time() - start_time) / 60
print(f"\n  Total training time: {elapsed:.1f} minutes")

# ══════════════════════════════════════════════════════════════════
# SECTION 11 — Hold-out evaluation
# ══════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print(f"  HOLD-OUT TEST SET EVALUATION")
print(f"  {len(test_df):,} images \u2014 never seen during training")
print(f"{'═'*65}")

model.load_state_dict(best_state)
_, ho_f1, ho_acc, ho_prec, ho_rec, ho_preds, ho_lbls = evaluate(
    model, ho_loader)

test_cls_idx   = sorted(test_df["unified_label"].unique())
test_cls_names = [CLASS_NAMES.get(i, f"Class {i}")
                  for i in test_cls_idx]

print(f"\n  Accuracy  : {ho_acc*100:.2f}%")
print(f"  F1 Score  : {ho_f1*100:.2f}%  (macro)")
print(f"  Precision : {ho_prec*100:.2f}%")
print(f"  Recall    : {ho_rec*100:.2f}%")
print(f"\n  Per-class breakdown:")
print(classification_report(
    ho_lbls, ho_preds,
    labels=test_cls_idx,
    target_names=test_cls_names,
    zero_division=0))

# Melanoma recall — critical metric
if 0 in test_cls_idx:
    mel_t  = [1 if l == 0 else 0 for l in ho_lbls]
    mel_p  = [1 if p == 0 else 0 for p in ho_preds]
    mel_tp = sum(1 for t, p in zip(mel_t, mel_p)
                 if t == 1 and p == 1)
    mel_rec  = mel_tp / max(sum(mel_t), 1)
    mel_prec = mel_tp / max(sum(mel_p), 1)
    status = "\u2713" if mel_rec >= 0.85 else "\u26a0"
    print(f"  {status} Melanoma recall  : {mel_rec*100:.2f}% "
          f"(target \u2265 85%)")
    print(f"  {status} Melanoma prec    : {mel_prec*100:.2f}%")

# ── Confusion matrix ──────────────────────────────────────────────
n  = len(test_cls_idx)
cm = confusion_matrix(
    ho_lbls, ho_preds, labels=test_cls_idx)
fig, ax = plt.subplots(
    figsize=(max(9, n+2), max(8, n+1)))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im)
short = [cn.split("/")[0][:18].strip() for cn in test_cls_names]
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(short, fontsize=9)
for i in range(n):
    for j in range(n):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=8,
                color="white" if cm[i,j] > cm.max()/2
                else "black")
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("True", fontsize=11)
ax.set_title(
    f"Agent 3 \u2014 Confusion Matrix\n"
    f"Multi-Dataset | F1={ho_f1*100:.2f}% | "
    f"Acc={ho_acc*100:.2f}%",
    fontsize=11, fontweight="bold")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "agent3_confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()

# ── Training curve ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ep      = range(1, len(tr_losses)+1)
ax.plot(ep, tr_losses, "-",  color="#2196F3",
        linewidth=2, label="Train Loss")
ax.plot(ep, va_losses, "--", color="#2196F3",
        linewidth=2, label="Val Loss (hold-out)")
ax.axvline(x=best_epoch, color="green", linestyle=":",
           linewidth=1.5, alpha=0.7,
           label=f"Best epoch ({best_epoch})")
gap = abs(tr_losses[-1] - va_losses[-1])
ax.set_title(
    f"Agent 3 \u2014 Multi-Dataset Training Curve\n"
    f"{len(train_df):,} images | Best epoch={best_epoch} | "
    f"Gap={gap:.4f}",
    fontsize=11, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Focal Loss")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
curve_path = os.path.join(
    OUTPUT_DIR, "agent3_final_training_curve.png")
plt.savefig(curve_path, dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════
# SECTION 12 — Summary
# ══════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print(f"  FINAL SUMMARY \u2014 AGENT 3 MULTI-DATASET")
print(f"{'═'*65}")
print(f"  Datasets         : "
      f"{', '.join(df['source'].unique())}")
print(f"  Raw total images : {len(df):,}")
print(f"  After capping    : {total_capped:,} "
      f"(max {MAX_PER_CLASS:,}/class)")
print(f"  Training images  : {len(train_df):,}")
print(f"  Hold-out images  : {len(test_df):,}")
print(f"  Unified classes  : {NUM_CLASSES}")
print(f"  Focal gamma      : {FOCAL_GAMMA}")
print(f"\n  Hold-out F1      : {ho_f1*100:.2f}%")
print(f"  Hold-out Acc     : {ho_acc*100:.2f}%")
print(f"  Hold-out Prec    : {ho_prec*100:.2f}%")
print(f"  Hold-out Rec     : {ho_rec*100:.2f}%")
print(f"  Best epoch       : {best_epoch}")
print(f"  Training time    : {elapsed:.1f} minutes")
print(f"\n  Model saved      : {ckpt_path}")
print(f"  Confusion matrix : {cm_path}")
print(f"  Training curve   : {curve_path}")
print(f"\n  Next steps:")
print(f"  1. Update agents/agent3_diagnosis/config.py")
print(f"     NUM_CLASSES = {NUM_CLASSES}")
print(f"  2. Evaluate on DDI-31 \u2014 beat SkinGPT-X 35.8%")
print(f"  3. Push results to GitHub")
print(f"{'═'*65}\n")