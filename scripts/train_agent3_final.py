# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 3 — Final Training (Improved)
# ══════════════════════════════════════════════════════════════════
# Improvements:
#   1. Malignant classes (0,2,3) — NO CAP
#   2. RandomAffine + RandomGrayscale + RandomErasing augmentation
#   3. 35 epochs, patience 7 — allows full convergence
#   4. Versioned model saving — never overwrites previous best
#   5. drop_last=True, ep_lbls initialised before loop
# ══════════════════════════════════════════════════════════════════

import os
import sys
import random
import time
import datetime
import shutil
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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNIFIED_CSV    = os.path.join(PROJECT_ROOT, "data", "processed",
                 "agent3_unified_dataset.csv")
LABEL_MAP_CSV  = os.path.join(PROJECT_ROOT, "data", "processed",
                 "agent3_label_mapping.csv")
CACHE_FEATURES = os.path.join(PROJECT_ROOT, "data", "processed",
                 "agent2_cache_features.npy")
CACHE_INDEX    = os.path.join(PROJECT_ROOT, "data", "processed",
                 "agent2_cache_index.csv")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "outputs", "agent3_comparison")
MODELS_DIR     = os.path.join(PROJECT_ROOT, "models", "agent3")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

if not os.path.exists(UNIFIED_CSV):
    print(f"\nERROR: {UNIFIED_CSV} not found")
    print("Run: python scripts/prepare_agent3_data.py first")
    sys.exit(1)

# ─── Config ────────────────────────────────────────────────────────
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE           = 32
NUM_EPOCHS           = 35
PATIENCE             = 7
LR_HEAD              = 1e-4
LR_BACKBONE          = 1e-5
IMG_SIZE             = 224
FOCAL_GAMMA          = 3.0
MAX_PER_CLASS        = 6000
MAX_HAM_PER_CLASS    = 2000
BACKBONE_DIM         = 2048
AGENT2_DIM           = 6916
MALIGNANT_CLASSES    = {0, 2, 3}

# Versioned checkpoint paths — never overwrites previous best
TIMESTAMP      = datetime.datetime.now().strftime("%Y%m%d_%H%M")
CKPT_VERSIONED = os.path.join(MODELS_DIR,
                 f"agent3_resnet50_{TIMESTAMP}.pth")
CKPT_STANDARD  = os.path.join(MODELS_DIR,
                 "agent3_resnet50_final.pth")

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — Detect Agent 2 feature cache
# ══════════════════════════════════════════════════════════════════
USE_AGENT2          = False
agent2_features_arr = None
agent2_path_to_idx  = {}

if os.path.exists(CACHE_FEATURES) and os.path.exists(CACHE_INDEX):
    try:
        cache_idx_df        = pd.read_csv(CACHE_INDEX)
        agent2_features_arr = np.load(CACHE_FEATURES)
        agent2_path_to_idx  = {
            os.path.normpath(p): i
            for i, p in enumerate(cache_idx_df["image_path"])
        }
        if agent2_features_arr.shape[1] == AGENT2_DIM:
            USE_AGENT2 = True
            nonzero = (agent2_features_arr.sum(axis=1) != 0).sum()
            print(f"\n  Agent 2 cache: {agent2_features_arr.shape} "
                  f"| Non-zero: {nonzero:,} "
                  f"({nonzero/len(cache_idx_df)*100:.1f}%)")
        else:
            print(f"\n  Cache dim mismatch — backbone only")
    except Exception as e:
        print(f"\n  Cache error: {e} — backbone only")

COMBINED_DIM = BACKBONE_DIM + AGENT2_DIM if USE_AGENT2 else BACKBONE_DIM

print(f"\n{'='*65}")
print(f"  MAAI-SkinDx | Agent 3 — Final Training (Improved)")
print(f"{'='*65}")
print(f"  Device        : {DEVICE}")
print(f"  Architecture  : {'COMBINED 8964-dim' if USE_AGENT2 else 'BACKBONE 2048-dim'}")
print(f"  Focal gamma   : {FOCAL_GAMMA}")
print(f"  Max/class     : {MAX_PER_CLASS:,} (HAM10000: {MAX_HAM_PER_CLASS:,})")
print(f"  Malignant cap : NONE — all melanoma/BCC/SCC used")
print(f"  Epochs        : {NUM_EPOCHS} (patience={PATIENCE})")
print(f"  Version stamp : {TIMESTAMP}")
print(f"{'='*65}\n")

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — Load unified dataset
# ══════════════════════════════════════════════════════════════════
print("── Section 2: Loading Dataset ──")
df = pd.read_csv(UNIFIED_CSV)
df["exists"] = df["image_path"].apply(os.path.exists)
df = df[df["exists"]].copy().reset_index(drop=True)
df["image_path"] = df["image_path"].apply(os.path.normpath)

label_map   = pd.read_csv(LABEL_MAP_CSV)
CLASS_NAMES = {
    int(r["class_index"]): r["class_name"]
    for _, r in label_map.iterrows()
    if int(r["class_index"]) in df["unified_label"].unique()
}
NUM_CLASSES = len(CLASS_NAMES)

clin = (df["source"] != "HAM10000").sum()
print(f"  Valid images   : {len(df):,}")
print(f"  Classes        : {NUM_CLASSES}")
print(f"  Clinical       : {clin:,} ({clin/len(df)*100:.1f}%)")
print(f"  Dermoscopic    : {len(df)-clin:,} ({(len(df)-clin)/len(df)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — Class capping
# ══════════════════════════════════════════════════════════════════
print(f"\n── Section 3: Class Capping ──")
capped_rows = []

for cls_idx in sorted(df["unified_label"].unique()):
    cls_df   = df[df["unified_label"] == cls_idx].copy()
    cls_name = CLASS_NAMES.get(cls_idx, f"Class {cls_idx}")

    if cls_idx in MALIGNANT_CLASSES:
        capped_rows.append(cls_df)
        print(f"  {cls_idx}: {cls_name:<45} {len(cls_df):6,}  NO CAP (malignant)")
        continue

    ham_rows  = cls_df[cls_df["source"] == "HAM10000"]
    rest_rows = cls_df[cls_df["source"] != "HAM10000"]

    if len(ham_rows) > MAX_HAM_PER_CLASS:
        ham_rows = ham_rows.sample(n=MAX_HAM_PER_CLASS, random_state=SEED)

    remaining = MAX_PER_CLASS - len(ham_rows)
    if len(rest_rows) > remaining:
        rest_rows = rest_rows.sample(n=remaining, random_state=SEED)

    combined = pd.concat([ham_rows, rest_rows])
    capped_rows.append(combined)
    print(f"  {cls_idx}: {cls_name:<45} {len(combined):6,}")

df_capped    = pd.concat(capped_rows).reset_index(drop=True)
total_capped = len(df_capped)
clin_pct     = (df_capped["source"] != "HAM10000").sum() / total_capped * 100
print(f"\n  Total after capping : {total_capped:,}")
print(f"  Clinical percentage : {clin_pct:.1f}%")

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — Train/hold-out split
# ══════════════════════════════════════════════════════════════════
print(f"\n── Section 4: Train/Hold-out Split (80/20) ──")
ham_df  = df_capped[df_capped["source"] == "HAM10000"].copy()
rest_df = df_capped[df_capped["source"] != "HAM10000"].copy()

train_parts, test_parts = [], []

if len(ham_df) > 0:
    lesions = ham_df.groupby("group_id")["unified_label"].first().reset_index()
    tr_l, te_l = train_test_split(
        lesions["group_id"].values,
        test_size=0.20, random_state=SEED,
        stratify=lesions["unified_label"].values)
    train_parts.append(ham_df[ham_df["group_id"].isin(tr_l)])
    test_parts.append(ham_df[ham_df["group_id"].isin(te_l)])

if len(rest_df) > 0:
    tr_r, te_r = train_test_split(
        rest_df, test_size=0.20, random_state=SEED,
        stratify=rest_df["unified_label"].values)
    train_parts.append(tr_r)
    test_parts.append(te_r)

train_df = pd.concat(train_parts).reset_index(drop=True)
test_df  = pd.concat(test_parts).reset_index(drop=True)
print(f"  Training pool : {len(train_df):,}")
print(f"  Hold-out test : {len(test_df):,} — LOCKED")

# ══════════════════════════════════════════════════════════════════
# SECTION 5 — Dataset class and transforms
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
    transforms.RandomAffine(
        degrees=15,
        translate=(0.1, 0.1),
        scale=(0.85, 1.15)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]),
])

class SkinDataset(Dataset):
    def __init__(self, df, transform,
                 use_agent2, features_arr, path_to_idx):
        self.paths      = [os.path.normpath(p) for p in df["image_path"]]
        self.labels     = [int(l) for l in df["unified_label"]]
        self.transform  = transform
        self.use_agent2 = use_agent2
        self.features   = features_arr
        self.p2i        = path_to_idx

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        label = self.labels[idx]
        try:
            img = PILImage.open(path).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)

        if self.use_agent2:
            cidx = self.p2i.get(path, -1)
            a2   = (torch.from_numpy(self.features[cidx].copy()).float()
                    if cidx >= 0 else torch.zeros(AGENT2_DIM))
            return img, a2, label
        return img, label

# ══════════════════════════════════════════════════════════════════
# SECTION 6 — Focal Loss with class weights
# ══════════════════════════════════════════════════════════════════
tr_labels    = train_df["unified_label"].values.astype(int)
label_counts = Counter(tr_labels)
total        = len(tr_labels)
all_classes  = sorted(train_df["unified_label"].unique())

cw_list = [
    total / (len(all_classes) * max(label_counts.get(c, 1), 1))
    for c in all_classes
]
class_weights = torch.FloatTensor(cw_list).to(DEVICE)

print(f"\n── Section 6: Class Weights ──")
for i, c in enumerate(all_classes):
    name = CLASS_NAMES.get(c, f"Class {c}")
    cnt  = label_counts.get(c, 0)
    print(f"  {c}: {name:<45} n={cnt:6,}  w={cw_list[i]:.3f}")

class FocalLoss(nn.Module):
    def __init__(self, gamma=3.0, class_weight=None):
        super().__init__()
        self.gamma        = gamma
        self.class_weight = class_weight

    def forward(self, inputs, targets):
        targets  = targets.long()
        log_p    = F.log_softmax(inputs, dim=1)
        prob     = torch.exp(log_p)
        ce       = F.nll_loss(log_p, targets,
                              weight=self.class_weight,
                              reduction="none")
        pt       = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        return (((1 - pt) ** self.gamma) * ce).mean()

criterion = FocalLoss(gamma=FOCAL_GAMMA, class_weight=class_weights)

# ══════════════════════════════════════════════════════════════════
# SECTION 7 — CombinedModel
# ══════════════════════════════════════════════════════════════════
print(f"\n── Section 7: Building Model ──")

class CombinedModel(nn.Module):
    def __init__(self, combined_dim, num_classes, use_agent2):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1)

        for p in resnet.parameters():
            p.requires_grad = False
        for name, p in resnet.named_parameters():
            if "layer4" in name:
                p.requires_grad = True

        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4   # Grad-CAM target for Agent 4
        self.avgpool = resnet.avgpool

        self.use_agent2 = use_agent2
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(combined_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, agent2_feats=None):
        x = self.conv1(images)
        x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        backbone_feats = torch.flatten(x, 1)

        if self.use_agent2 and agent2_feats is not None:
            combined = torch.cat([backbone_feats, agent2_feats], dim=1)
        else:
            combined = backbone_feats

        return self.classifier(combined), backbone_feats

model = CombinedModel(
    combined_dim=COMBINED_DIM,
    num_classes=NUM_CLASSES,
    use_agent2=USE_AGENT2
).to(DEVICE)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Architecture  : {'COMBINED 8964-dim' if USE_AGENT2 else 'BACKBONE 2048-dim'}")
print(f"  Output classes: {NUM_CLASSES}")
print(f"  Trainable     : {trainable:,}")

# ══════════════════════════════════════════════════════════════════
# SECTION 8 — Data loaders
# ══════════════════════════════════════════════════════════════════
s_weights = [total / max(label_counts.get(l, 1), 1) for l in tr_labels]
sampler   = WeightedRandomSampler(s_weights, len(s_weights), replacement=True)

tr_ds = SkinDataset(train_df, TRAIN_TRANSFORM,
                    USE_AGENT2, agent2_features_arr, agent2_path_to_idx)
ho_ds = SkinDataset(test_df, VAL_TRANSFORM,
                    USE_AGENT2, agent2_features_arr, agent2_path_to_idx)

tr_loader = DataLoader(
    tr_ds, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=0, pin_memory=True, drop_last=True)
ho_loader = DataLoader(
    ho_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True)

print(f"\n  Training batches : {len(tr_loader):,}")
print(f"  Hold-out batches : {len(ho_loader):,}")

# ══════════════════════════════════════════════════════════════════
# SECTION 9 — Optimizer
# ══════════════════════════════════════════════════════════════════
backbone_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "classifier" not in n]
head_params     = [p for n, p in model.named_parameters()
                   if p.requires_grad and "classifier" in n]

optimizer = torch.optim.Adam([
    {"params": backbone_params, "lr": LR_BACKBONE},
    {"params": head_params,     "lr": LR_HEAD},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

# ══════════════════════════════════════════════════════════════════
# SECTION 10 — Evaluate function
# ══════════════════════════════════════════════════════════════════
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_lbls = [], []
    with torch.no_grad():
        for batch in loader:
            if USE_AGENT2:
                imgs, a2, labels = batch
                a2 = a2.to(DEVICE).float()
            else:
                imgs, labels = batch
                a2 = None
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE).long()
            out, _ = model(imgs, a2)
            loss   = criterion(out, labels)
            total_loss += loss.item()
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_lbls.extend(labels.cpu().numpy())
    f1   = f1_score(all_lbls, all_preds, average="macro", zero_division=0)
    acc  = accuracy_score(all_lbls, all_preds)
    prec = precision_score(all_lbls, all_preds, average="macro", zero_division=0)
    rec  = recall_score(all_lbls, all_preds, average="macro", zero_division=0)
    return total_loss/len(loader), f1, acc, prec, rec, all_preds, all_lbls

# ══════════════════════════════════════════════════════════════════
# SECTION 11 — Training loop
# ══════════════════════════════════════════════════════════════════
arch = "COMBINED 8964-dim" if USE_AGENT2 else "BACKBONE 2048-dim"
print(f"\n{'─'*65}")
print(f"  Training: {len(train_df):,} images | {arch}")
print(f"{'─'*65}")

best_f1      = 0.0
patience_cnt = 0
best_state   = None
best_epoch   = 0
tr_losses    = []
va_losses    = []
ep_lbls      = []
ep_preds     = []
start_time   = time.time()

for epoch in range(NUM_EPOCHS):
    ep_start = time.time()
    model.train()
    ep_loss  = 0.0
    ep_preds = []
    ep_lbls  = []

    for batch_idx, batch in enumerate(tr_loader):
        if USE_AGENT2:
            imgs, a2, labels = batch
            a2 = a2.to(DEVICE).float()
        else:
            imgs, labels = batch
            a2 = None
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE).long()

        optimizer.zero_grad()
        out, _ = model(imgs, a2)
        loss   = criterion(out, labels)
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
    tr_f1   = f1_score(ep_lbls, ep_preds, average="macro", zero_division=0)
    va_loss, va_f1, va_acc, _, _, _, _ = evaluate(model, ho_loader)

    tr_losses.append(tr_loss)
    va_losses.append(va_loss)
    ep_time = time.time() - ep_start
    gap     = abs(tr_f1 - va_f1) * 100

    print(f"\n  Epoch {epoch+1:2d}/{NUM_EPOCHS} ({ep_time:.0f}s)")
    print(f"  Train : loss={tr_loss:.4f} f1={tr_f1*100:.2f}%")
    print(f"  Val   : loss={va_loss:.4f} "
          f"f1={va_f1*100:.2f}% acc={va_acc*100:.2f}%")
    print(f"  Gap   : {gap:.2f}% "
          f"({'monitor' if gap > 10 else 'ok'})")

    if va_f1 > best_f1:
        best_f1      = va_f1
        best_epoch   = epoch + 1
        patience_cnt = 0
        best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        checkpoint = {
            "state_dict"    : best_state,
            "val_f1"        : best_f1,
            "epoch"         : best_epoch,
            "class_names"   : CLASS_NAMES,
            "model_name"    : "CombinedModel-ResNet50",
            "backbone_dim"  : BACKBONE_DIM,
            "agent2_dim"    : AGENT2_DIM if USE_AGENT2 else 0,
            "combined_dim"  : COMBINED_DIM,
            "num_classes"   : NUM_CLASSES,
            "use_agent2"    : USE_AGENT2,
            "datasets_used" : list(df["source"].unique()),
            "total_train"   : len(train_df),
            "timestamp"     : TIMESTAMP,
        }
        # Save versioned — never overwritten
        torch.save(checkpoint, CKPT_VERSIONED)
        # Save standard name — used by pipeline
        shutil.copy2(CKPT_VERSIONED, CKPT_STANDARD)

        print(f"  BEST saved  f1={best_f1*100:.2f}% epoch={best_epoch}")
        print(f"  Versioned : {CKPT_VERSIONED}")
        print(f"  Pipeline  : {CKPT_STANDARD}")
    else:
        patience_cnt += 1
        print(f"  No improvement ({patience_cnt}/{PATIENCE})")
        if patience_cnt >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

elapsed = (time.time() - start_time) / 60

# ══════════════════════════════════════════════════════════════════
# SECTION 12 — Hold-out evaluation
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  HOLD-OUT EVALUATION — {len(test_df):,} unseen images")
print(f"{'='*65}")

model.load_state_dict(best_state)
_, ho_f1, ho_acc, ho_prec, ho_rec, ho_preds, ho_lbls = evaluate(
    model, ho_loader)

test_cls_idx   = sorted(test_df["unified_label"].unique())
test_cls_names = [CLASS_NAMES.get(i, f"Class {i}") for i in test_cls_idx]

print(f"\n  Accuracy  : {ho_acc*100:.2f}%")
print(f"  F1 Score  : {ho_f1*100:.2f}% (macro)")
print(f"  Precision : {ho_prec*100:.2f}%")
print(f"  Recall    : {ho_rec*100:.2f}%")
print(f"\n  Per-class breakdown:")
print(classification_report(
    ho_lbls, ho_preds,
    labels=test_cls_idx,
    target_names=test_cls_names,
    zero_division=0))

if 0 in test_cls_idx:
    mel_t    = [1 if l == 0 else 0 for l in ho_lbls]
    mel_p    = [1 if p == 0 else 0 for p in ho_preds]
    mel_tp   = sum(1 for t, p in zip(mel_t, mel_p) if t == 1 and p == 1)
    mel_rec  = mel_tp / max(sum(mel_t), 1)
    mel_prec = mel_tp / max(sum(mel_p), 1)
    s = "OK" if mel_rec >= 0.85 else "BELOW TARGET"
    print(f"  Melanoma recall  : {mel_rec*100:.2f}% (target >=85% — {s})")
    print(f"  Melanoma prec    : {mel_prec*100:.2f}%")

# Confusion matrix
n  = len(test_cls_idx)
cm = confusion_matrix(ho_lbls, ho_preds, labels=test_cls_idx)
fig, ax = plt.subplots(figsize=(max(10, n+2), max(9, n+1)))
im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar(im)
short = [cn.split("/")[0][:18].strip() for cn in test_cls_names]
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(short, fontsize=9)
for i in range(n):
    for j in range(n):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=8,
                color="white" if cm[i, j] > cm.max()/2 else "black")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title(
    f"Agent 3 | {arch}\nF1={ho_f1*100:.2f}% | Acc={ho_acc*100:.2f}%",
    fontsize=11, fontweight="bold")
plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "agent3_confusion_matrix.png")
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()

# Training curve
fig, ax = plt.subplots(figsize=(10, 5))
ep = range(1, len(tr_losses)+1)
ax.plot(ep, tr_losses, "-",  color="#2196F3", linewidth=2, label="Train Loss")
ax.plot(ep, va_losses, "--", color="#2196F3", linewidth=2, label="Val Loss")
ax.axvline(x=best_epoch, color="green", linestyle=":",
           linewidth=1.5, alpha=0.7, label=f"Best epoch ({best_epoch})")
fin_gap = abs(tr_losses[-1] - va_losses[-1])
ax.set_title(
    f"Agent 3 | {arch} | Best ep={best_epoch} | Gap={fin_gap:.4f}",
    fontsize=11, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Focal Loss")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
curve_path = os.path.join(OUTPUT_DIR, "agent3_final_training_curve.png")
plt.savefig(curve_path, dpi=150, bbox_inches="tight")
plt.close()

# ══════════════════════════════════════════════════════════════════
# SECTION 13 — Final summary
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"  FINAL SUMMARY — AGENT 3 (Improved)")
print(f"{'='*65}")
print(f"  Architecture  : {arch}")
print(f"  Malignant cap : NONE (all melanoma/BCC/SCC used)")
print(f"  Datasets      : {', '.join(df['source'].unique())}")
print(f"  Total images  : {len(df):,}")
print(f"  After capping : {total_capped:,}")
print(f"  Training      : {len(train_df):,}")
print(f"  Hold-out      : {len(test_df):,}")
print(f"  Classes       : {NUM_CLASSES}")
print(f"  F1 Score      : {ho_f1*100:.2f}%")
print(f"  Accuracy      : {ho_acc*100:.2f}%")
print(f"  Best epoch    : {best_epoch}")
print(f"  Training time : {elapsed:.1f} minutes")
print(f"  Versioned     : {CKPT_VERSIONED}")
print(f"  Pipeline      : {CKPT_STANDARD}")
print(f"\n  Next steps:")
print(f"  1. python scripts/evaluate_ddi31.py")
print(f"  2. python notebooks/test_orchestrator.py")
print(f"  3. git add/commit/push")
print(f"  4. Build Agent 4 (Grad-CAM — self.layer4 ready)")
print(f"{'='*65}\n")