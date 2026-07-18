# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 2 — Dermoscopic Stream Model Comparison
# ══════════════════════════════════════════════════════════════════
# Four-model comparison for dermoscopic feature extractor.
# Three Vision Transformers + one CNN baseline.
#
# Models:
#   1. PanDerm ViT-B/16     (specialized, 2M+ dermoscopic images)
#   2. ViT-B/16 ImageNet    (same architecture, general pretraining)
#   3. DINOv2 ViT-B/14      (strong self-supervised ViT, Meta AI)
#   4. ResNet-50 ImageNet   (CNN baseline for reference only)
#
# Evaluation: Linear probe AND k-NN probe (5-fold CV)
#   Linear probe  → standard, but unfair for ViTs (reference only)
#   k-NN probe    → correct method for ViT feature spaces
#   PRIMARY winner selected by k-NN F1
#
# Dataset : HAM10000 (10,015 images, 7 disease classes)
# Split   : StratifiedGroupKFold by lesion_id (no data leakage)
# Caching : Features saved to disk — no re-extraction if rerun
#
# FIX vs previous version:
#   DINOv2 requires 518x518 input — separate transform applied
#   All other models use 224x224
# ══════════════════════════════════════════════════════════════════

import os
import random
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import pickle
from PIL import Image as PILImage
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import timm

# ─── Reproducibility ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ─── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAM_PART1    = os.path.join(PROJECT_ROOT, "data", "raw", "ham10000",
               "HAM10000_images_part_1")
HAM_PART2    = os.path.join(PROJECT_ROOT, "data", "raw", "ham10000",
               "HAM10000_images_part_2")
HAM_META     = os.path.join(PROJECT_ROOT, "data", "raw", "ham10000",
               "HAM10000_metadata")
PANDERM_CKPT = os.path.join(PROJECT_ROOT, "models", "panderm",
               "panderm_bb_data6_checkpoint-499.pth")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "outputs", "agent2_features")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs",
               "agent2_derm_comparison")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,   exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'═'*65}")
print(f"  MAAI-SkinDx | Agent 2 — Dermoscopic Stream Comparison")
print(f"{'═'*65}")
print(f"  Device   : {DEVICE}")
print(f"  Dataset  : HAM10000 (10,015 images, 7 classes)")
print(f"  Split    : StratifiedGroupKFold by lesion_id")
print(f"  Models   : PanDerm, ViT-B/16, DINOv2 ViT-B/14, ResNet-50")
print(f"  Probes   : Linear (reference) + k-NN (primary for ViTs)")
print(f"  Winner   : selected by k-NN F1")
print(f"{'═'*65}\n")

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — Load HAM10000
# ══════════════════════════════════════════════════════════════════
print("── Section 1: Loading HAM10000 ──")

meta = pd.read_csv(HAM_META, sep=',')

def find_ham_image(image_id):
    for folder in [HAM_PART1, HAM_PART2]:
        p = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(p):
            return p
    return None

meta['image_path'] = meta['image_id'].apply(find_ham_image)
meta = meta[meta['image_path'].notna()].copy().reset_index(drop=True)

le            = LabelEncoder()
meta['label'] = le.fit_transform(meta['dx'])
ham_paths     = list(meta['image_path'])
ham_labels    = meta['label'].values
ham_groups    = meta['lesion_id'].values

print(f"  Total images   : {len(meta)}")
print(f"  Unique lesions : {meta['lesion_id'].nunique()}")
print(f"  Classes        : {list(le.classes_)}")
print(f"  Distribution   :")
for dx, cnt in meta['dx'].value_counts().items():
    bar = '█' * (cnt // 200)
    print(f"    {dx:8s} {cnt:5d}  {bar}")

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — Image transforms
# Two transforms: standard 224 for most models, 518 for DINOv2
# ══════════════════════════════════════════════════════════════════

# Standard transform — PanDerm, ViT-B/16 ImageNet, ResNet-50
TRANSFORM_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# DINOv2 transform — must be 518x518 (native training resolution)
TRANSFORM_518 = transforms.Compose([
    transforms.Resize((518, 518)),
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
            # Return zero tensor of correct size on error
            h = self.transform.transforms[0].size
            if isinstance(h, (list, tuple)):
                h = h[0]
            return torch.zeros(3, h, h), self.labels[idx], idx

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — Model builders (all frozen)
# ══════════════════════════════════════════════════════════════════

def build_panderm():
    """PanDerm ViT-B/16 — local checkpoint, no download needed."""
    try:
        model = timm.create_model(
            'vit_base_patch16_224', pretrained=False, num_classes=0)
        ckpt  = torch.load(
            PANDERM_CKPT, map_location='cpu', weights_only=False)
        state = ckpt.get('model', ckpt.get('state_dict', ckpt))
        state = {k.replace('module.', ''): v
                 for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        print(f"  PanDerm ViT-B/16 loaded — dim: 768")
        return model.to(DEVICE), 768
    except Exception as e:
        print(f"  PanDerm load failed: {e}")
        return None, 768

def build_vit_imagenet():
    """
    ViT-B/16 pretrained on ImageNet.
    Same architecture as PanDerm — only pretraining differs.
    Key comparison: does dermoscopic pretraining add value?
    """
    try:
        model = timm.create_model(
            'vit_base_patch16_224.augreg2_in21k_ft_in1k',
            pretrained=True, num_classes=0)
        print(f"  ViT-B/16 ImageNet loaded — dim: 768")
    except Exception:
        try:
            model = timm.create_model(
                'vit_base_patch16_224',
                pretrained=True, num_classes=0)
            print(f"  ViT-B/16 ImageNet loaded — dim: 768")
        except Exception as e:
            print(f"  ViT-B/16 failed: {e}")
            return None, 768

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model.to(DEVICE), 768

def build_dinov2():
    """
    DINOv2 ViT-B/14 — Meta AI self-supervised.
    IMPORTANT: requires 518x518 input — use TRANSFORM_518.
    State-of-the-art general ViT features.
    """
    try:
        model = timm.create_model(
            'vit_base_patch14_dinov2.lvd142m',
            pretrained=True, num_classes=0,
            img_size=518)
        print(f"  DINOv2 ViT-B/14 loaded via timm — dim: 768  "
              f"[input: 518x518]")
    except Exception:
        try:
            print(f"  Trying torch.hub for DINOv2...")
            model = torch.hub.load(
                'facebookresearch/dinov2',
                'dinov2_vitb14',
                trust_repo=True)
            print(f"  DINOv2 ViT-B/14 loaded via torch.hub — dim: 768  "
                  f"[input: 518x518]")
        except Exception as e:
            print(f"  DINOv2 failed: {e}")
            return None, 768

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model.to(DEVICE), 768

def build_resnet50():
    """ResNet-50 pretrained on ImageNet — CNN baseline."""
    model = models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    model.fc = nn.Identity()
    model.eval()
    print(f"  ResNet-50 ImageNet loaded — dim: 2048")
    return model.to(DEVICE), 2048

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — Feature extraction with disk caching
# Accepts transform parameter — DINOv2 uses TRANSFORM_518
# ══════════════════════════════════════════════════════════════════

def extract_features(model, paths, labels, cache_name,
                     feat_dim, transform, batch_size=32):
    save_path = os.path.join(
        FEATURES_DIR,
        f"{cache_name.replace(' ','_').replace('/','_')}.pkl"
    )

    if os.path.exists(save_path):
        print(f"  Loading cached: {cache_name}")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    print(f"  Extracting {len(paths)} images — {cache_name}...")
    ds     = ImageDataset(paths, labels, transform)
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=0,
                        pin_memory=True)

    all_feats  = np.zeros((len(paths), feat_dim), dtype=np.float32)
    all_labels = np.array(labels, dtype=np.int64)

    model.eval()
    with torch.no_grad():
        for batch_idx, (imgs, lbls, idxs) in enumerate(loader):
            imgs  = imgs.to(DEVICE)
            feats = model(imgs)
            if isinstance(feats, (list, tuple)):
                feats = feats[0]
            feats = feats.view(feats.size(0), -1)
            all_feats[idxs.numpy()] = feats.cpu().numpy()
            if (batch_idx + 1) % 30 == 0:
                done = (batch_idx + 1) * batch_size
                print(f"    {min(done, len(paths))}/{len(paths)}")

    result = {'features': all_feats, 'labels': all_labels}
    with open(save_path, 'wb') as f:
        pickle.dump(result, f)
    print(f"  Saved: {save_path}")
    return result

# ══════════════════════════════════════════════════════════════════
# SECTION 5 — Dual probe evaluation (linear + k-NN)
# ══════════════════════════════════════════════════════════════════

def evaluate_model(features, labels, groups,
                   model_name, n_splits=5):
    cv = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=SEED)

    lin_f1s, knn_f1s   = [], []
    lin_accs, knn_accs = [], []

    print(f"\n  ── {model_name} ──")
    print(f"  {'Fold':<5} {'Linear F1':<13} {'k-NN F1':<13} "
          f"{'Linear Acc':<13} {'k-NN Acc'}")
    print(f"  {'─'*5} {'─'*13} {'─'*13} {'─'*13} {'─'*10}")

    for fold, (tr_idx, va_idx) in enumerate(
        cv.split(features, labels, groups=groups)
    ):
        X_tr = features[tr_idx]
        X_va = features[va_idx]
        y_tr = labels[tr_idx]
        y_va = labels[va_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_va   = scaler.transform(X_va)

        # Linear probe
        lin = LogisticRegression(
            class_weight='balanced', max_iter=1000,
            random_state=SEED, solver='lbfgs',
            multi_class='auto')
        lin.fit(X_tr, y_tr)
        lin_pred = lin.predict(X_va)
        lin_f1   = f1_score(y_va, lin_pred,
                            average='macro', zero_division=0)
        lin_acc  = accuracy_score(y_va, lin_pred)
        lin_f1s.append(lin_f1)
        lin_accs.append(lin_acc)

        # k-NN probe — PRIMARY for ViTs
        knn = KNeighborsClassifier(
            n_neighbors=10,
            weights='distance',
            metric='cosine',
            n_jobs=-1)
        knn.fit(X_tr, y_tr)
        knn_pred = knn.predict(X_va)
        knn_f1   = f1_score(y_va, knn_pred,
                            average='macro', zero_division=0)
        knn_acc  = accuracy_score(y_va, knn_pred)
        knn_f1s.append(knn_f1)
        knn_accs.append(knn_acc)

        print(f"  [{fold+1}/{n_splits}]  "
              f"{lin_f1*100:.2f}%        "
              f"{knn_f1*100:.2f}%        "
              f"{lin_acc*100:.2f}%        "
              f"{knn_acc*100:.2f}%")

    result = {
        'model_name'    : model_name,
        'linear_f1'     : float(np.mean(lin_f1s)),
        'linear_f1_std' : float(np.std(lin_f1s)),
        'linear_acc'    : float(np.mean(lin_accs)),
        'knn_f1'        : float(np.mean(knn_f1s)),
        'knn_f1_std'    : float(np.std(knn_f1s)),
        'knn_acc'       : float(np.mean(knn_accs)),
    }

    print(f"\n  Linear probe : "
          f"F1={result['linear_f1']*100:.2f}%"
          f" ± {result['linear_f1_std']*100:.2f}%  "
          f"Acc={result['linear_acc']*100:.2f}%")
    print(f"  k-NN probe   : "
          f"F1={result['knn_f1']*100:.2f}%"
          f" ± {result['knn_f1_std']*100:.2f}%  "
          f"Acc={result['knn_acc']*100:.2f}%"
          f"  ← PRIMARY")

    return result

# ══════════════════════════════════════════════════════════════════
# SECTION 6 — Run all four models
# model_defs now includes transform per model
# ══════════════════════════════════════════════════════════════════
print("\n" + "═"*65)
print("  RUNNING FOUR-MODEL COMPARISON")
print("  ViT vs ViT vs ViT vs CNN baseline")
print("═"*65)

# Each entry: (name, builder, feat_dim, cache_name, transform)
model_defs = [
    ("PanDerm ViT-B/16",
     build_panderm, 768, "PanDerm_HAM10000", TRANSFORM_224),

    ("ViT-B/16 ImageNet",
     build_vit_imagenet, 768, "ViTB16_ImageNet_HAM10000", TRANSFORM_224),

    ("DINOv2 ViT-B/14",
     build_dinov2, 768, "DINOv2_HAM10000", TRANSFORM_518),
    # DINOv2 uses TRANSFORM_518 — trained at 518x518 natively

    ("ResNet-50 ImageNet",
     build_resnet50, 2048, "ResNet50_HAM10000", TRANSFORM_224),
]

results = {}

for model_name, builder, feat_dim, cache_name, tf in model_defs:
    print(f"\n── {model_name} ──")
    model, dim = builder()

    if model is None:
        save_path = os.path.join(
            FEATURES_DIR,
            f"{cache_name.replace(' ','_').replace('/','_')}.pkl"
        )
        if os.path.exists(save_path):
            print(f"  Model load failed but features cached — "
                  f"using cached features")
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
            results[model_name] = evaluate_model(
                data['features'], data['labels'],
                ham_groups, model_name)
        else:
            print(f"  Skipping {model_name} — "
                  f"no model and no cached features")
        continue

    data = extract_features(
        model, ham_paths, ham_labels,
        cache_name, dim, tf, batch_size=32)
    del model
    torch.cuda.empty_cache()

    results[model_name] = evaluate_model(
        data['features'], data['labels'],
        ham_groups, model_name)

# ══════════════════════════════════════════════════════════════════
# SECTION 7 — Results table
# ══════════════════════════════════════════════════════════════════
if not results:
    print("No models evaluated. Check downloads and rerun.")
    exit()

knn_winner = max(results, key=lambda n: results[n]['knn_f1'])
lin_winner = max(results, key=lambda n: results[n]['linear_f1'])

print(f"\n{'═'*65}")
print(f"  RESULTS — DERMOSCOPIC STREAM")
print(f"{'═'*65}")
print(f"\n  {'Model':<24} {'Linear F1':>16} {'k-NN F1':>16}")
print(f"  {'─'*24} {'─'*16} {'─'*16}")
for name, r in results.items():
    tag = "  ← k-NN WINNER" if name == knn_winner else ""
    print(f"  {name:<24} "
          f"{r['linear_f1']*100:.2f}%±{r['linear_f1_std']*100:.2f}%   "
          f"{r['knn_f1']*100:.2f}%±{r['knn_f1_std']*100:.2f}%"
          f"{tag}")

print(f"\n  PRIMARY winner (k-NN)    : {knn_winner}")
print(f"  REFERENCE winner (Linear): {lin_winner}")

if knn_winner == "PanDerm ViT-B/16":
    print(f"\n  Scientific conclusion:")
    print(f"  PanDerm wins — dermoscopic medical pretraining IS")
    print(f"  beneficial over general ViT pretraining.")
    print(f"  Justifies keeping PanDerm in Agent 2.")
elif knn_winner == "ViT-B/16 ImageNet":
    print(f"\n  Scientific conclusion:")
    print(f"  ViT-B/16 ImageNet wins — same architecture as PanDerm")
    print(f"  but general pretraining performs better on this task.")
elif knn_winner == "DINOv2 ViT-B/14":
    print(f"\n  Scientific conclusion:")
    print(f"  DINOv2 wins — powerful self-supervised general features")
    print(f"  outperform specialized medical pretraining on HAM10000.")
    print(f"  Use DINOv2 in Agent 2 dermoscopic stream.")
else:
    print(f"\n  Scientific conclusion:")
    print(f"  ResNet-50 wins on both probes — CNN features outperform")
    print(f"  all ViTs on this evaluation of HAM10000.")

# ══════════════════════════════════════════════════════════════════
# SECTION 8 — Plots
# ══════════════════════════════════════════════════════════════════
model_names = list(results.keys())
lin_f1s     = [results[n]['linear_f1']*100 for n in model_names]
knn_f1s     = [results[n]['knn_f1']*100    for n in model_names]
lin_stds    = [results[n]['linear_f1_std']*100 for n in model_names]
knn_stds    = [results[n]['knn_f1_std']*100    for n in model_names]

color_map = {
    "PanDerm ViT-B/16"  : "#1565C0",
    "ViT-B/16 ImageNet" : "#42A5F5",
    "DINOv2 ViT-B/14"   : "#7E57C2",
    "ResNet-50 ImageNet": "#66BB6A",
}
bar_colors = [color_map.get(n, "#999999") for n in model_names]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(
    "MAAI-SkinDx | Agent 2 — Dermoscopic Stream Comparison\n"
    "HAM10000  |  5-Fold StratifiedGroupKFold by lesion_id  |  "
    "k-NN is PRIMARY metric for ViT features",
    fontsize=12, fontweight='bold')

x = np.arange(len(model_names))

for ax, f1s, stds, title, primary in [
    (axes[0], lin_f1s, lin_stds,
     "Linear Probe F1\n(reference — disadvantages ViTs)", False),
    (axes[1], knn_f1s, knn_stds,
     "k-NN Probe F1 (cosine, k=10)\n★ PRIMARY metric", True),
]:
    bars = ax.bar(x, f1s, color=bar_colors,
                  edgecolor='black', linewidth=0.8, width=0.55)
    ax.errorbar(x, f1s, yerr=stds,
                fmt='none', color='black', capsize=6, linewidth=2)

    wi = f1s.index(max(f1s))
    ax.text(wi, f1s[wi] + stds[wi] + 0.5,
            "★ WINNER", ha='center', va='bottom',
            color='green', fontweight='bold', fontsize=11)

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.2,
                f"{val:.1f}%",
                ha='center', va='bottom',
                fontweight='bold', fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [n.replace(' ', '\n') for n in model_names], fontsize=9)
    ax.set_title(title, fontweight='bold',
                 color='darkblue' if primary else 'gray')
    ax.set_ylabel("Mean F1 Score (%) — Macro")
    ax.set_ylim([max(0, min(f1s)-15), min(100, max(f1s)+10)])
    ax.grid(True, alpha=0.3, axis='y')

legend_patches = [
    plt.Rectangle((0,0), 1, 1,
                  fc=color_map.get(n, "#999999"),
                  ec='black', lw=0.8)
    for n in model_names
]
fig.legend(legend_patches, model_names,
           loc='lower center', ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=[0, 0.06, 1, 1])
plot_path = os.path.join(OUTPUT_DIR, "derm_stream_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Plot saved: {plot_path}")

# ══════════════════════════════════════════════════════════════════
# SECTION 9 — Final summary
# ══════════════════════════════════════════════════════════════════
wr = results[knn_winner]

print(f"\n{'═'*65}")
print(f"  FINAL SUMMARY — DERMOSCOPIC STREAM")
print(f"{'═'*65}")
print(f"\n  Models evaluated : {len(results)}/4")
print(f"  Primary metric   : k-NN F1 (correct for ViT features)")
print(f"\n  {'Model':<24} {'k-NN F1':>14} {'Linear F1':>14}")
print(f"  {'─'*24} {'─'*14} {'─'*14}")
for name, r in results.items():
    tag = "  ← WINNER" if name == knn_winner else ""
    print(f"  {name:<24} "
          f"{r['knn_f1']*100:.2f}%±{r['knn_f1_std']*100:.2f}%   "
          f"{r['linear_f1']*100:.2f}%±{r['linear_f1_std']*100:.2f}%"
          f"{tag}")

print(f"\n  Winner    : {knn_winner}")
print(f"  k-NN F1   : {wr['knn_f1']*100:.2f}%"
      f" ± {wr['knn_f1_std']*100:.2f}%")
print(f"  Linear F1 : {wr['linear_f1']*100:.2f}%"
      f" ± {wr['linear_f1_std']*100:.2f}%")
print(f"\n  Plot: {plot_path}")
print(f"{'═'*65}\n")