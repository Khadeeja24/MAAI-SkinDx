# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 2 — Feature Extractor Comparison
# ══════════════════════════════════════════════════════════════════
# Evaluates feature extractors for dermoscopic and clinical streams.
# Method: Linear probe with 5-fold cross validation.
# No fine-tuning — all models stay completely frozen.
# Features saved to disk — no re-extraction if script restarts.
#
# Dermoscopic stream (HAM10000, 7 disease classes):
#   PanDerm ViT-B/16       (specialized, 2M+ dermoscopic images)
#   ResNet-50 ImageNet     (standard medical baseline)
#   DenseNet-121 ImageNet  (validated medical imaging baseline)
#
# Clinical stream (Fitzpatrick-17k, 3 partition classes):
#   Derm Foundation BiT    (specialized, 4.5M clinical images)
#   ResNet-50 ImageNet     (consistent baseline)
#   EfficientNet-B4 ImageNet (strong clinical imaging baseline)
#
# HAM10000 split: StratifiedGroupKFold by lesion_id
#   Prevents data leakage — same lesion never in both train and val
# Fitzpatrick split: StratifiedKFold by disease class
# Class imbalance: class_weight='balanced' in logistic regression
# Winner: highest mean macro F1 across 5 folds
# ══════════════════════════════════════════════════════════════════

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import pickle
from pathlib import Path
from PIL import Image as PILImage
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder

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
HAM_PART1     = os.path.join(PROJECT_ROOT, "data", "raw", "ham10000", "HAM10000_images_part_1")
HAM_PART2     = os.path.join(PROJECT_ROOT, "data", "raw", "ham10000", "HAM10000_images_part_2")
HAM_META      = os.path.join(PROJECT_ROOT, "data", "raw", "ham10000", "HAM10000_metadata")
FITZ_CSV      = os.path.join(PROJECT_ROOT, "data", "raw", "fitzpatrick17k", "fitzpatrick17k.csv")
FITZ_IMG_BASE = os.path.join(PROJECT_ROOT, "data", "raw", "fitzpatrick17k", "fitzpatrick_3groups")
PANDERM_CKPT  = os.path.join(PROJECT_ROOT, "models", "panderm", "panderm_bb_data6_checkpoint-499.pth")
FEATURES_DIR  = os.path.join(PROJECT_ROOT, "outputs", "agent2_features")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "outputs", "agent2_comparison")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,   exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'═'*65}")
print(f"  MAAI-SkinDx | Agent 2 — Feature Extractor Comparison")
print(f"{'═'*65}")
print(f"  Device  : {DEVICE}")
print(f"  Seed    : {SEED}")
print(f"  Method  : Linear probe + 5-fold stratified CV")
print(f"  HAM10000 split: StratifiedGroupKFold by lesion_id")
print(f"  Fitzpatrick  : StratifiedKFold by class")
print(f"{'═'*65}\n")

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — Load and prepare datasets
# ══════════════════════════════════════════════════════════════════
print("── Section 1: Preparing datasets ──\n")

# ── HAM10000 ───────────────────────────────────────────────────────
print("  Loading HAM10000...")
meta = pd.read_csv(HAM_META, sep=',')

def find_ham_image(image_id):
    for folder in [HAM_PART1, HAM_PART2]:
        p = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(p):
            return p
    return None

meta['image_path'] = meta['image_id'].apply(find_ham_image)
meta = meta[meta['image_path'].notna()].copy().reset_index(drop=True)

le_ham = LabelEncoder()
meta['label'] = le_ham.fit_transform(meta['dx'])

ham_paths     = list(meta['image_path'])
ham_labels    = meta['label'].values
ham_lesion_ids= meta['lesion_id'].values   # used as groups in CV

print(f"  Total images     : {len(meta)}")
print(f"  Unique lesions   : {meta['lesion_id'].nunique()}")
print(f"  Disease classes  : {list(le_ham.classes_)}")
print(f"  Class counts     :")
for dx, cnt in meta['dx'].value_counts().items():
    print(f"    {dx:10s}: {cnt}")

# ── Fitzpatrick-17k ────────────────────────────────────────────────
print(f"\n  Loading Fitzpatrick-17k...")
fitz_csv = pd.read_csv(FITZ_CSV)

rows = []
for group_folder in ['group1_light', 'group2_medium', 'group3_dark']:
    folder = os.path.join(FITZ_IMG_BASE, group_folder)
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                rows.append({
                    'md5hash'   : os.path.splitext(f)[0],
                    'image_path': os.path.join(folder, f),
                })

fitz_df = pd.DataFrame(rows)
fitz_df = fitz_df.merge(
    fitz_csv[['md5hash', 'three_partition_label']],
    on='md5hash', how='inner'
)
fitz_df = fitz_df[fitz_df['three_partition_label'].notna()].copy()
fitz_df = fitz_df[fitz_df['three_partition_label'] != ''].copy()
fitz_df = fitz_df.reset_index(drop=True)

le_fitz = LabelEncoder()
fitz_df['label'] = le_fitz.fit_transform(fitz_df['three_partition_label'])

fitz_paths  = list(fitz_df['image_path'])
fitz_labels = fitz_df['label'].values

print(f"  Total images matched : {len(fitz_df)}")
print(f"  Classes (3-partition): {list(le_fitz.classes_)}")
print(f"  Class counts:")
for cls, cnt in fitz_df['three_partition_label'].value_counts().items():
    print(f"    {cls}: {cnt}")

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — Dataset and preprocessing
# ══════════════════════════════════════════════════════════════════
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = PILImage.open(self.paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx], idx
        except Exception:
            return torch.zeros(3, 224, 224), self.labels[idx], idx

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — Model builders (all frozen)
# ══════════════════════════════════════════════════════════════════

def build_panderm():
    import timm
    model = timm.create_model(
        'vit_base_patch16_224', pretrained=False, num_classes=0)
    ckpt  = torch.load(PANDERM_CKPT, map_location='cpu', weights_only=False)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    print(f"  PanDerm loaded from checkpoint — feature dim: 768")
    return model.to(DEVICE), 768

def build_resnet50(tag=""):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Identity()
    model.eval()
    print(f"  ResNet-50{tag} loaded — feature dim: 2048")
    return model.to(DEVICE), 2048

def build_densenet121():
    model = models.densenet121(
        weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.classifier = nn.Identity()
    model.eval()
    print(f"  DenseNet-121 loaded — feature dim: 1024")
    return model.to(DEVICE), 1024

def build_efficientnet_b4():
    model   = models.efficientnet_b4(
        weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    feat_dim = model.classifier[1].in_features   # 1792 for B4
    model.classifier = nn.Identity()
    model.eval()
    print(f"  EfficientNet-B4 loaded — feature dim: {feat_dim}")
    return model.to(DEVICE), feat_dim

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — PyTorch feature extraction (batched, GPU)
# ══════════════════════════════════════════════════════════════════

def extract_torch_features(model, paths, labels,
                            cache_name, feat_dim, batch_size=32):
    save_path = os.path.join(
        FEATURES_DIR, f"{cache_name.replace(' ', '_')}.pkl")

    if os.path.exists(save_path):
        print(f"  Loading cached: {cache_name}")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    print(f"  Extracting {len(paths)} images — {cache_name}...")
    ds     = ImageDataset(paths, labels, TRANSFORM)
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=0,
                        pin_memory=True)

    all_features = np.zeros((len(paths), feat_dim), dtype=np.float32)
    all_labels   = np.array(labels, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, lbls, idxs) in enumerate(loader):
            imgs  = imgs.to(DEVICE)
            feats = model(imgs)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            feats = feats.view(feats.size(0), -1)
            all_features[idxs.numpy()] = feats.cpu().numpy()
            if (batch_idx + 1) % 50 == 0:
                print(f"    Batch {batch_idx+1}/{len(loader)}")

    result = {'features': all_features, 'labels': all_labels}
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"  Saved: {save_path}")
    return result

# ══════════════════════════════════════════════════════════════════
# SECTION 5 — Derm Foundation extraction (TF, sequential)
# ══════════════════════════════════════════════════════════════════

def extract_derm_foundation(paths, labels):
    save_path = os.path.join(FEATURES_DIR, "DermFoundation.pkl")

    if os.path.exists(save_path):
        print(f"  Loading cached: Derm Foundation")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    print(f"  Extracting {len(paths)} images — Derm Foundation...")
    print(f"  Note: processes one image at a time — "
          f"~{len(paths)//60} minutes estimated")

    import tensorflow as tf
    from huggingface_hub import snapshot_download
    import io

    model_path = snapshot_download("google/derm-foundation")
    model      = tf.saved_model.load(model_path)
    infer      = model.signatures["serving_default"]

    features   = np.zeros((len(paths), 6144), dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int64)
    failed     = 0

    for i, path in enumerate(paths):
        try:
            pil_img = PILImage.open(path).convert('RGB')
            buf     = io.BytesIO()
            pil_img.save(buf, format='PNG')
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'image/encoded': tf.train.Feature(
                        bytes_list=tf.train.BytesList(
                            value=[buf.getvalue()]))
                })
            ).SerializeToString()
            out           = infer(inputs=tf.constant(
                [example], dtype=tf.string))
            features[i]   = out['embedding'].numpy().squeeze()
        except Exception:
            features[i]   = np.zeros(6144)
            failed        += 1

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{len(paths)} — failed: {failed}")

    result = {'features': features, 'labels': labels_arr}
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"  Saved: {save_path}  (failed: {failed})")
    return result

# ══════════════════════════════════════════════════════════════════
# SECTION 6 — Linear probe evaluation
# ══════════════════════════════════════════════════════════════════

def linear_probe(features, labels, model_name,
                 groups=None, n_splits=5):
    """
    5-fold cross validation linear probe.
    groups: if provided, uses StratifiedGroupKFold (HAM10000)
            to prevent same lesion appearing in train and val.
    """
    if groups is not None:
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = cv.split(features, labels, groups=groups)
        cv_type = "StratifiedGroupKFold (lesion-aware)"
    else:
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=SEED)
        splits = cv.split(features, labels)
        cv_type = "StratifiedKFold"

    print(f"\n  Linear probe — {model_name}")
    print(f"  CV type: {cv_type}")
    print(f"  {'Fold':<6} {'Accuracy':<12} {'Precision':<12} "
          f"{'Recall':<10} {'F1'}")
    print(f"  {'─'*6} {'─'*12} {'─'*12} {'─'*10} {'─'*8}")

    fold_acc, fold_prec, fold_rec, fold_f1 = [], [], [], []

    for fold, (tr_idx, va_idx) in enumerate(splits):
        X_tr, X_va = features[tr_idx], features[va_idx]
        y_tr, y_va = labels[tr_idx],   labels[va_idx]

        clf = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=SEED,
            solver='lbfgs',
            multi_class='auto'
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_va)

        acc  = accuracy_score(y_va,  y_pred)
        prec = precision_score(y_va, y_pred,
               average='macro', zero_division=0)
        rec  = recall_score(y_va, y_pred,
               average='macro', zero_division=0)
        f1   = f1_score(y_va, y_pred,
               average='macro', zero_division=0)

        fold_acc.append(acc)
        fold_prec.append(prec)
        fold_rec.append(rec)
        fold_f1.append(f1)

        print(f"  [{fold+1}/{n_splits}]  "
              f"{acc*100:.2f}%       "
              f"{prec*100:.2f}%       "
              f"{rec*100:.2f}%    "
              f"{f1*100:.2f}%")

    result = {
        'model_name'    : model_name,
        'mean_accuracy' : float(np.mean(fold_acc)),
        'mean_precision': float(np.mean(fold_prec)),
        'mean_recall'   : float(np.mean(fold_rec)),
        'mean_f1'       : float(np.mean(fold_f1)),
        'std_f1'        : float(np.std(fold_f1)),
    }

    print(f"\n  Average: "
          f"Acc={result['mean_accuracy']*100:.2f}%  "
          f"F1={result['mean_f1']*100:.2f}%"
          f" ± {result['std_f1']*100:.2f}%")

    return result

# ══════════════════════════════════════════════════════════════════
# SECTION 7 — Dermoscopic stream (HAM10000)
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  DERMOSCOPIC STREAM — HAM10000")
print("  7 disease classes, lesion-aware 5-fold CV")
print("═"*65)

derm_results = {}

# PanDerm
print(f"\n── PanDerm ViT-B/16 ──")
model, fdim = build_panderm()
data = extract_torch_features(
    model, ham_paths, ham_labels,
    "PanDerm_HAM10000", fdim)
del model; torch.cuda.empty_cache()
derm_results["PanDerm ViT-B/16"] = linear_probe(
    data['features'], data['labels'],
    "PanDerm ViT-B/16",
    groups=ham_lesion_ids)

# ResNet-50
print(f"\n── ResNet-50 ImageNet ──")
model, fdim = build_resnet50()
data = extract_torch_features(
    model, ham_paths, ham_labels,
    "ResNet50_HAM10000", fdim)
del model; torch.cuda.empty_cache()
derm_results["ResNet-50 ImageNet"] = linear_probe(
    data['features'], data['labels'],
    "ResNet-50 ImageNet",
    groups=ham_lesion_ids)

# DenseNet-121
print(f"\n── DenseNet-121 ImageNet ──")
model, fdim = build_densenet121()
data = extract_torch_features(
    model, ham_paths, ham_labels,
    "DenseNet121_HAM10000", fdim)
del model; torch.cuda.empty_cache()
derm_results["DenseNet-121 ImageNet"] = linear_probe(
    data['features'], data['labels'],
    "DenseNet-121 ImageNet",
    groups=ham_lesion_ids)

# ══════════════════════════════════════════════════════════════════
# SECTION 8 — Clinical stream (Fitzpatrick-17k)
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  CLINICAL STREAM — Fitzpatrick-17k")
print("  3 partition classes (benign/malignant/non-neoplastic)")
print("  Stratified 5-fold CV (no lesion grouping needed)")
print("═"*65)

clin_results = {}

# ResNet-50 (clinical)
print(f"\n── ResNet-50 ImageNet (Clinical) ──")
model, fdim = build_resnet50(" (Clinical)")
data = extract_torch_features(
    model, fitz_paths, fitz_labels,
    "ResNet50_Fitzpatrick", fdim)
del model; torch.cuda.empty_cache()
clin_results["ResNet-50 ImageNet"] = linear_probe(
    data['features'], data['labels'],
    "ResNet-50 ImageNet")

# EfficientNet-B4 (clinical)
print(f"\n── EfficientNet-B4 ImageNet (Clinical) ──")
model, fdim = build_efficientnet_b4()
data = extract_torch_features(
    model, fitz_paths, fitz_labels,
    "EfficientNetB4_Fitzpatrick", fdim)
del model; torch.cuda.empty_cache()
clin_results["EfficientNet-B4 ImageNet"] = linear_probe(
    data['features'], data['labels'],
    "EfficientNet-B4 ImageNet")

# Derm Foundation (clinical)
print(f"\n── Derm Foundation BiT (Clinical) ──")
data = extract_derm_foundation(fitz_paths, fitz_labels)
clin_results["Derm Foundation BiT"] = linear_probe(
    data['features'], data['labels'],
    "Derm Foundation BiT")

# ══════════════════════════════════════════════════════════════════
# SECTION 9 — Results and plots
# ══════════════════════════════════════════════════════════════════
derm_winner = max(derm_results, key=lambda n: derm_results[n]['mean_f1'])
clin_winner = max(clin_results, key=lambda n: clin_results[n]['mean_f1'])

print(f"\n{'═'*65}")
print(f"  DERMOSCOPIC STREAM RESULTS (HAM10000)")
print(f"{'═'*65}")
print(f"  {'Model':<26} {'Accuracy':<12} {'F1':<12} {'Std Dev'}")
print(f"  {'─'*26} {'─'*12} {'─'*12} {'─'*10}")
for name, r in derm_results.items():
    marker = "  ← WINNER" if name == derm_winner else ""
    print(f"  {name:<26} "
          f"{r['mean_accuracy']*100:.2f}%      "
          f"{r['mean_f1']*100:.2f}%      "
          f"±{r['std_f1']*100:.2f}%"
          f"{marker}")

print(f"\n{'═'*65}")
print(f"  CLINICAL STREAM RESULTS (Fitzpatrick-17k)")
print(f"{'═'*65}")
print(f"  {'Model':<26} {'Accuracy':<12} {'F1':<12} {'Std Dev'}")
print(f"  {'─'*26} {'─'*12} {'─'*12} {'─'*10}")
for name, r in clin_results.items():
    marker = "  ← WINNER" if name == clin_winner else ""
    print(f"  {name:<26} "
          f"{r['mean_accuracy']*100:.2f}%      "
          f"{r['mean_f1']*100:.2f}%      "
          f"±{r['std_f1']*100:.2f}%"
          f"{marker}")

# Bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "MAAI-SkinDx | Agent 2 — Feature Extractor Comparison\n"
    "Linear Probe  |  5-Fold Cross Validation",
    fontsize=13, fontweight='bold')

for ax, results, title, bar_colors in [
    (axes[0], derm_results,
     "Dermoscopic Stream\n(HAM10000, 7 classes)",
     ["#2196F3", "#4CAF50", "#FF9800"]),
    (axes[1], clin_results,
     "Clinical Stream\n(Fitzpatrick-17k, 3 classes)",
     ["#4CAF50", "#FF9800", "#9C27B0"]),
]:
    names = list(results.keys())
    f1s   = [results[n]['mean_f1'] * 100 for n in names]
    stds  = [results[n]['std_f1']  * 100 for n in names]

    bars = ax.bar(range(len(names)), f1s,
                  color=bar_colors[:len(names)],
                  edgecolor='black', linewidth=0.8, width=0.5)
    ax.errorbar(range(len(names)), f1s, yerr=stds,
                fmt='none', color='black', capsize=6, linewidth=2)

    wi = f1s.index(max(f1s))
    ax.text(wi, f1s[wi] + stds[wi] + 0.5,
            "★ WINNER", ha='center', va='bottom',
            color='green', fontweight='bold', fontsize=10)

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.2,
                f"{val:.2f}%",
                ha='center', va='bottom',
                fontweight='bold', fontsize=10)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(
        [n.replace(' ', '\n') for n in names], fontsize=9)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel("Mean F1 Score (%)")
    ax.set_ylim([max(0, min(f1s)-10), min(100, max(f1s)+8)])
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "agent2_model_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Plot saved: {plot_path}")

# ══════════════════════════════════════════════════════════════════
# SECTION 10 — Final summary
# ══════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print(f"  FINAL SUMMARY")
print(f"{'═'*65}")
print(f"\n  Dermoscopic Stream Winner : {derm_winner}")
dr = derm_results[derm_winner]
print(f"    Mean F1   : {dr['mean_f1']*100:.2f}% ± {dr['std_f1']*100:.2f}%")
print(f"    Accuracy  : {dr['mean_accuracy']*100:.2f}%")
print(f"\n  Clinical Stream Winner    : {clin_winner}")
cr = clin_results[clin_winner]
print(f"    Mean F1   : {cr['mean_f1']*100:.2f}% ± {cr['std_f1']*100:.2f}%")
print(f"    Accuracy  : {cr['mean_accuracy']*100:.2f}%")
print(f"\n  Next steps:")
print(f"  1. Update Agent 2 streams with winning models")
print(f"  2. Remove losing model streams from Agent 2")
print(f"  3. Rerun test_orchestrator.py")
print(f"{'═'*65}\n")