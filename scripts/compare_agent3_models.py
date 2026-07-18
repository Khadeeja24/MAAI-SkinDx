# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 3 — Disease Classification Model Comparison
# ══════════════════════════════════════════════════════════════════
# Compares three CNN backbones fine-tuned for skin disease diagnosis.
#
# Models compared:
#   1. EfficientNet-B3  — primary candidate, SOTA on HAM10000
#   2. ResNet-50        — consistent baseline across all agents
#   3. DenseNet-121     — medically validated (CheXNet, ISIC winners)
#
# Dataset  : HAM10000 (10,015 images, 7 disease classes)
# Split    : StratifiedGroupKFold by lesion_id (no data leakage)
# Imbalance: WeightedRandomSampler + Focal Loss (gamma=2)
# Fine-tune: Backbone last blocks + new classification head
# Winner   : selected by mean macro F1 across 5 folds
# ══════════════════════════════════════════════════════════════════

import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
from PIL import Image as PILImage
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HAM_PART1    = os.path.join(PROJECT_ROOT, "data", "raw", "ham10000", "HAM10000_images_part_1")
HAM_PART2    = os.path.join(PROJECT_ROOT, "data", "raw", "ham10000", "HAM10000_images_part_2")
HAM_META     = os.path.join(PROJECT_ROOT, "data", "raw", "ham10000", "HAM10000_metadata")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "agent3_comparison")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models", "agent3")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE   = 32
NUM_EPOCHS   = 15
PATIENCE     = 5
LR_HEAD      = 1e-4
LR_BACKBONE  = 1e-5
N_SPLITS     = 5
NUM_CLASSES  = 7
IMG_SIZE     = 224
FOCAL_GAMMA  = 2.0

print(f"\n{'═'*65}")
print(f"  MAAI-SkinDx | Agent 3 — Model Comparison")
print(f"{'═'*65}")
print(f"  Device      : {DEVICE}")
print(f"  Models      : EfficientNet-B3, ResNet-50, DenseNet-121")
print(f"  Dataset     : HAM10000 (7 classes)")
print(f"  CV folds    : {N_SPLITS}")
print(f"  Max epochs  : {NUM_EPOCHS} (early stop patience={PATIENCE})")
print(f"  Batch size  : {BATCH_SIZE}")
print(f"  Focal gamma : {FOCAL_GAMMA}")
print(f"{'═'*65}\n")

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — Load HAM10000
# ══════════════════════════════════════════════════════════════════
print("── Section 1: Loading HAM10000 ──")

meta = pd.read_csv(HAM_META, sep=',')

def find_image(image_id):
    for folder in [HAM_PART1, HAM_PART2]:
        p = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(p):
            return p
    return None

meta['image_path'] = meta['image_id'].apply(find_image)
meta = meta[meta['image_path'].notna()].copy().reset_index(drop=True)

le            = LabelEncoder()
meta['label'] = le.fit_transform(meta['dx'])
CLASS_NAMES   = list(le.classes_)

print(f"  Total images   : {len(meta)}")
print(f"  Unique lesions : {meta['lesion_id'].nunique()}")
print(f"  Classes        : {CLASS_NAMES}")
print(f"  Distribution:")
for cls, cnt in meta['dx'].value_counts().items():
    pct = cnt / len(meta) * 100
    bar = '█' * (cnt // 200)
    print(f"    {cls:8s} {cnt:5d} ({pct:4.1f}%)  {bar}")

# Hold-out split by lesion_id
lesion_df = meta.groupby('lesion_id').agg(
    label=('label', 'first')).reset_index()

train_lesions, holdout_lesions = train_test_split(
    lesion_df['lesion_id'].values,
    test_size=0.20, random_state=SEED,
    stratify=lesion_df['label'].values
)

meta_train   = meta[meta['lesion_id'].isin(train_lesions)].copy().reset_index(drop=True)
meta_holdout = meta[meta['lesion_id'].isin(holdout_lesions)].copy().reset_index(drop=True)

print(f"\n  Training pool : {len(meta_train)} images ({meta_train['lesion_id'].nunique()} lesions)")
print(f"  Hold-out test : {len(meta_holdout)} images — LOCKED")
print(f"\n  Hold-out class distribution:")
for cls, cnt in meta_holdout['dx'].value_counts().items():
    print(f"    {cls:8s}: {cnt}")

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — Dataset and transforms
# ══════════════════════════════════════════════════════════════════

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class SkinDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = list(paths)
        self.labels    = [int(l) for l in labels]  # always int
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = PILImage.open(self.paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, self.labels[idx]
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), self.labels[idx]

def make_weighted_sampler(labels):
    labels   = [int(l) for l in labels]
    counts   = Counter(labels)
    total    = len(labels)
    weights  = [total / counts[l] for l in labels]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — Focal Loss
# ══════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets    = targets.long()  # always cast to long
        log_prob   = F.log_softmax(inputs, dim=1)
        prob       = torch.exp(log_prob)
        ce         = F.nll_loss(log_prob, targets, reduction='none')
        pt         = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_loss = ((1 - pt) ** self.gamma) * ce
        return focal_loss.mean()

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — Model builders
# ══════════════════════════════════════════════════════════════════

def build_efficientnet_b3():
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if 'features.7' in name or 'features.8' in name:
            p.requires_grad = True
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    return model.to(DEVICE), "EfficientNet-B3", in_features

def build_resnet50():
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if 'layer4' in name:
            p.requires_grad = True
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    return model.to(DEVICE), "ResNet-50", in_features

def build_densenet121():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if 'denseblock4' in name or 'norm5' in name:
            p.requires_grad = True
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES)
    )
    return model.to(DEVICE), "DenseNet-121", in_features

def get_optimizer(model):
    head_names      = ['classifier', 'fc']
    head_params     = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(h in name for h in head_names):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return torch.optim.Adam([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params,     'lr': LR_HEAD},
    ])

# ══════════════════════════════════════════════════════════════════
# SECTION 5 — Train and evaluate
# ══════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs   = imgs.to(DEVICE)
        labels = labels.to(DEVICE).long()  # ← fix: always long

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"      Batch {batch_idx+1}/{len(loader)} loss={loss.item():.4f}")

    return (total_loss / len(loader),
            f1_score(all_labels, all_preds, average='macro', zero_division=0),
            accuracy_score(all_labels, all_preds))

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    criterion = FocalLoss(gamma=FOCAL_GAMMA)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(DEVICE)
            labels = labels.to(DEVICE).long()  # ← fix: always long
            outputs    = model(imgs)
            loss       = criterion(outputs, labels)
            total_loss += loss.item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (total_loss / len(loader),
            f1_score(all_labels, all_preds, average='macro', zero_division=0),
            accuracy_score(all_labels, all_preds),
            precision_score(all_labels, all_preds, average='macro', zero_division=0),
            recall_score(all_labels, all_preds, average='macro', zero_division=0),
            all_preds, all_labels)

# ══════════════════════════════════════════════════════════════════
# SECTION 6 — K-fold cross validation
# ══════════════════════════════════════════════════════════════════

def run_kfold(builder_fn):
    model, model_name, feat_dim = builder_fn()
    print(f"\n  Feature dim      : {feat_dim}")
    print(f"  Trainable params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    init_path = os.path.join(MODELS_DIR, f"_init_{model_name}.pth")
    torch.save(model.state_dict(), init_path)
    del model
    torch.cuda.empty_cache()

    sgkf   = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    paths  = meta_train['image_path'].values
    labels = meta_train['label'].values.astype(int)
    groups = meta_train['lesion_id'].values

    fold_f1s, fold_accs, fold_precs, fold_recs = [], [], [], []
    all_tr_loss, all_va_loss = [], []

    print(f"\n  {'Fold':<5} {'Val F1':<12} {'Val Acc':<12} {'Precision':<12} {'Recall':<10} {'Best Ep'}")
    print(f"  {'─'*5} {'─'*12} {'─'*12} {'─'*12} {'─'*10} {'─'*7}")

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(paths, labels, groups=groups)):
        print(f"\n── Fold {fold+1}/{N_SPLITS} ──")
        torch.manual_seed(SEED + fold)

        model, _, _ = builder_fn()
        state = torch.load(init_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(state)

        tr_paths,  tr_labels = paths[tr_idx],  labels[tr_idx]
        va_paths,  va_labels = paths[va_idx],  labels[va_idx]

        tr_ds     = SkinDataset(tr_paths, tr_labels, TRAIN_TRANSFORM)
        va_ds     = SkinDataset(va_paths, va_labels, VAL_TRANSFORM)
        sampler   = make_weighted_sampler(tr_labels)
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

        optimizer = get_optimizer(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)
        criterion = FocalLoss(gamma=FOCAL_GAMMA)

        best_val_f1, patience_cnt, best_epoch, best_state = 0.0, 0, 0, None
        fold_tr, fold_va = [], []

        for epoch in range(NUM_EPOCHS):
            print(f"\n  [Fold {fold+1} | Epoch {epoch+1}/{NUM_EPOCHS}]")
            tr_loss, tr_f1, tr_acc = train_one_epoch(model, tr_loader, optimizer, criterion)
            va_loss, va_f1, va_acc, va_prec, va_rec, _, _ = evaluate(model, va_loader)
            scheduler.step()

            fold_tr.append(tr_loss)
            fold_va.append(va_loss)

            print(f"  Train: loss={tr_loss:.4f} f1={tr_f1*100:.2f}%")
            print(f"  Val  : loss={va_loss:.4f} f1={va_f1*100:.2f}% acc={va_acc*100:.2f}%")

            if va_f1 > best_val_f1:
                best_val_f1  = va_f1
                best_epoch   = epoch + 1
                patience_cnt = 0
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                ckpt = os.path.join(MODELS_DIR,
                    f"{model_name.replace('-','').replace(' ','_')}_fold{fold+1}.pth")
                torch.save({'state_dict': best_state, 'val_f1': best_val_f1,
                            'epoch': best_epoch, 'classes': CLASS_NAMES,
                            'model_name': model_name}, ckpt)
                print(f"  ✓ Best saved (f1={best_val_f1*100:.2f}%)")
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        all_tr_loss.append(fold_tr)
        all_va_loss.append(fold_va)

        model.load_state_dict(best_state)
        _, va_f1, va_acc, va_prec, va_rec, _, _ = evaluate(model, va_loader)

        fold_f1s.append(va_f1)
        fold_accs.append(va_acc)
        fold_precs.append(va_prec)
        fold_recs.append(va_rec)

        print(f"\n  [{fold+1}/{N_SPLITS}] F1={va_f1*100:.2f}% Acc={va_acc*100:.2f}% "
              f"Prec={va_prec*100:.2f}% Rec={va_rec*100:.2f}% BestEp={best_epoch}")

        del model
        torch.cuda.empty_cache()

    if os.path.exists(init_path):
        os.remove(init_path)

    return {
        'model_name' : model_name,
        'feat_dim'   : feat_dim,
        'mean_f1'    : float(np.mean(fold_f1s)),
        'std_f1'     : float(np.std(fold_f1s)),
        'mean_acc'   : float(np.mean(fold_accs)),
        'mean_prec'  : float(np.mean(fold_precs)),
        'mean_rec'   : float(np.mean(fold_recs)),
        'fold_f1s'   : fold_f1s,
        'tr_losses'  : all_tr_loss,
        'va_losses'  : all_va_loss,
    }

# ══════════════════════════════════════════════════════════════════
# SECTION 7 — Run all three models
# ══════════════════════════════════════════════════════════════════

BUILDERS    = [build_efficientnet_b3, build_resnet50, build_densenet121]
all_results = {}
start_time  = time.time()

for builder in BUILDERS:
    m, name, d = builder()
    del m; torch.cuda.empty_cache()

    print(f"\n{'═'*65}")
    print(f"  RUNNING: {name}")
    print(f"{'═'*65}")

    result = run_kfold(builder)
    all_results[name] = result

    print(f"\n  ── {name} Complete ──")
    print(f"  Mean F1  : {result['mean_f1']*100:.2f}% ± {result['std_f1']*100:.2f}%")
    print(f"  Mean Acc : {result['mean_acc']*100:.2f}%")

elapsed = (time.time() - start_time) / 3600
print(f"\n  Total training time: {elapsed:.2f} hours")

# ══════════════════════════════════════════════════════════════════
# SECTION 8 — Results
# ══════════════════════════════════════════════════════════════════
winner = max(all_results, key=lambda n: all_results[n]['mean_f1'])

print(f"\n{'═'*65}")
print(f"  K-FOLD RESULTS")
print(f"{'═'*65}")
print(f"\n  {'Model':<22} {'F1 mean±std':<22} {'Accuracy':<12} {'Precision':<12} Recall")
print(f"  {'─'*22} {'─'*22} {'─'*12} {'─'*12} {'─'*10}")
for name, r in all_results.items():
    tag = "  ← WINNER" if name == winner else ""
    print(f"  {name:<22} {r['mean_f1']*100:.2f}%±{r['std_f1']*100:.2f}%           "
          f"{r['mean_acc']*100:.2f}%      {r['mean_prec']*100:.2f}%      "
          f"{r['mean_rec']*100:.2f}%{tag}")

# ── Plots ──────────────────────────────────────────────────────────
COLORS = {"EfficientNet-B3": "#2196F3", "ResNet-50": "#4CAF50", "DenseNet-121": "#FF9800"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("MAAI-SkinDx | Agent 3 — Training vs Validation Loss\n5-Fold CV | HAM10000",
             fontsize=13, fontweight='bold')

for ax, (model_name, r) in zip(axes, all_results.items()):
    max_len = max(len(c) for c in r['tr_losses'])
    def pad(curves): return [c + [c[-1]] * (max_len - len(c)) for c in curves]
    tr_arr  = np.array(pad(r['tr_losses']))
    va_arr  = np.array(pad(r['va_losses']))
    epochs  = range(1, max_len + 1)
    c       = COLORS[model_name]
    ax.plot(epochs, tr_arr.mean(0), '-',  color=c, linewidth=2, label='Train')
    ax.plot(epochs, va_arr.mean(0), '--', color=c, linewidth=2, label='Val')
    ax.fill_between(epochs, va_arr.mean(0)-va_arr.std(0),
                    va_arr.mean(0)+va_arr.std(0), alpha=0.15, color=c)
    gap = abs(float(tr_arr.mean(0)[-1]) - float(va_arr.mean(0)[-1]))
    verdict = "⚠ Overfit" if gap > 0.05 else "✓ OK"
    tag = " ★" if model_name == winner else ""
    ax.set_title(f"{model_name}{tag}\nF1={r['mean_f1']*100:.2f}%  Gap={gap:.4f}  {verdict}",
                 fontsize=10, fontweight='bold')
    ax.set_xlabel("Epoch"); ax.set_ylabel("Focal Loss")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "agent3_overfitting_check.png"), dpi=150, bbox_inches='tight')
plt.close()

names  = list(all_results.keys())
f1s    = [all_results[n]['mean_f1']*100 for n in names]
stds   = [all_results[n]['std_f1']*100  for n in names]
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(names)), f1s, color=[COLORS[n] for n in names],
              edgecolor='black', linewidth=0.8, width=0.5)
ax.errorbar(range(len(names)), f1s, yerr=stds, fmt='none', color='black', capsize=6, linewidth=2)
wi = f1s.index(max(f1s))
ax.text(wi, f1s[wi]+stds[wi]+0.3, "★ WINNER", ha='center', va='bottom',
        color='green', fontweight='bold', fontsize=12)
for bar, val in zip(bars, f1s):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
            f"{val:.2f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel("Mean Macro F1 (%)"); ax.grid(True, alpha=0.3, axis='y')
ax.set_title("MAAI-SkinDx | Agent 3 — Model Comparison\nHAM10000 | 5-Fold StratifiedGroupKFold",
             fontsize=12, fontweight='bold')
ax.set_ylim([max(0, min(f1s)-10), min(100, max(f1s)+8)])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "agent3_model_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

# ══════════════════════════════════════════════════════════════════
# SECTION 9 — Final summary
# ══════════════════════════════════════════════════════════════════
wr = all_results[winner]
print(f"\n{'═'*65}")
print(f"  FINAL SUMMARY")
print(f"{'═'*65}")
print(f"\n  {'Model':<22} {'F1':>10} {'Std':>8}")
print(f"  {'─'*22} {'─'*10} {'─'*8}")
for name, r in all_results.items():
    tag = "  ← WINNER" if name == winner else ""
    print(f"  {name:<22} {r['mean_f1']*100:.2f}%     ±{r['std_f1']*100:.2f}%{tag}")
print(f"\n  Winner      : {winner}")
print(f"  Mean CV F1  : {wr['mean_f1']*100:.2f}% ± {wr['std_f1']*100:.2f}%")
print(f"  Mean CV Acc : {wr['mean_acc']*100:.2f}%")
print(f"  Feature dim : {wr['feat_dim']}")
print(f"\n  Next steps:")
print(f"  1. Run scripts/train_agent3_final.py to retrain {winner} on full training pool")
print(f"  2. Evaluate on {len(meta_holdout)}-image held-out test set")
print(f"  3. Build agents/agent3_diagnosis/agent.py")
print(f"  4. Connect to Orchestrator")
print(f"  5. Evaluate on DDI-31 — beat SkinGPT-X 35.8%")
print(f"\n  Checkpoints : {MODELS_DIR}")
print(f"  Plots       : {OUTPUT_DIR}")
print(f"{'═'*65}\n")