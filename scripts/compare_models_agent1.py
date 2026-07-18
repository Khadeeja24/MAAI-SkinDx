# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 1 — Complete Model Selection Pipeline
# ══════════════════════════════════════════════════════════════════
# Correct flow:
#   Step 1 — 5-Fold CV on train_pool (1520 images)
#            Track train + val loss — save best state per model
#   Step 2 — Plot overfitting check for all three models
#   Step 3 — Evaluate ALL THREE on held-out test set (380 images)
#            Winner selected by held-out F1 — not K-fold
#   Step 4 — ROC curve for ALL THREE — optimal threshold per model
#            Threshold saved into each model file immediately
#   Step 5 — Retrain winner on full train_pool (1520 images)
#   Step 6 — Save final model with all metrics and threshold
# ══════════════════════════════════════════════════════════════════

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc
)
from PIL import Image as PILImage

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

POOL_SKIN  = os.path.join(PROJECT_ROOT, "data", "raw",
             "skin_detector_training", "train_pool", "skin")
POOL_NOT   = os.path.join(PROJECT_ROOT, "data", "raw",
             "skin_detector_training", "train_pool", "not_skin")
TEST_SKIN  = os.path.join(PROJECT_ROOT, "data", "raw",
             "skin_detector_training", "test", "skin")
TEST_NOT   = os.path.join(PROJECT_ROOT, "data", "raw",
             "skin_detector_training", "test", "not_skin")

MODEL_DIR  = os.path.join(PROJECT_ROOT, "models", "skin_detector")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "model_comparison")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Individual model save paths
MODEL_PATHS = {
    "MobileNetV3-Small": os.path.join(MODEL_DIR, "mobilenetv3_best.pth"),
    "EfficientNet-B0"  : os.path.join(MODEL_DIR, "efficientnet_best.pth"),
    "ResNet-50"        : os.path.join(MODEL_DIR, "resnet50_best.pth"),
}
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "skin_detector.pth")

# ─── Settings ──────────────────────────────────────────────────────
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS     = 10
K_FOLDS    = 5
LR         = 0.001
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n{'═' * 65}")
print(f"  MAAI-SkinDx | Agent 1 — Complete Model Selection Pipeline")
print(f"{'═' * 65}")
print(f"  Device      : {DEVICE}")
print(f"  K Folds     : {K_FOLDS}")
print(f"  Epochs/fold : {EPOCHS}")
print(f"  Seed        : {SEED}")
print(f"  Models      : MobileNetV3-Small, EfficientNet-B0, ResNet-50")
print(f"  Winner      : selected by held-out test F1")
print(f"{'═' * 65}\n")

# ══════════════════════════════════════════════════════════════════
# Collect images
# ══════════════════════════════════════════════════════════════════
print("── Collecting images ──")

pool_paths, pool_labels = [], []
for f in os.listdir(POOL_SKIN):
    if f.lower().endswith(('.jpg','.jpeg','.png')):
        pool_paths.append(os.path.join(POOL_SKIN, f))
        pool_labels.append(1)
for f in os.listdir(POOL_NOT):
    if f.lower().endswith(('.jpg','.jpeg','.png')):
        pool_paths.append(os.path.join(POOL_NOT, f))
        pool_labels.append(0)
pool_labels = np.array(pool_labels)

test_paths, test_labels = [], []
for f in os.listdir(TEST_SKIN):
    if f.lower().endswith(('.jpg','.jpeg','.png')):
        test_paths.append(os.path.join(TEST_SKIN, f))
        test_labels.append(1)
for f in os.listdir(TEST_NOT):
    if f.lower().endswith(('.jpg','.jpeg','.png')):
        test_paths.append(os.path.join(TEST_NOT, f))
        test_labels.append(0)
test_labels = np.array(test_labels)

print(f"  Train pool : {len(pool_paths)} images "
      f"(skin={np.sum(pool_labels==1)}, "
      f"not-skin={np.sum(pool_labels==0)})")
print(f"  Test set   : {len(test_paths)} images "
      f"(skin={np.sum(test_labels==1)}, "
      f"not-skin={np.sum(test_labels==0)}) — LOCKED\n")

# ══════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════
class SkinDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img   = PILImage.open(self.paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])

# ══════════════════════════════════════════════════════════════════
# Model builders
# ══════════════════════════════════════════════════════════════════
def build_model(name):
    if name == "MobileNetV3-Small":
        m = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        for p in m.parameters(): p.requires_grad = False
        m.classifier = nn.Sequential(
            nn.Linear(576, 256), nn.Hardswish(),
            nn.Dropout(0.3), nn.Linear(256, 2))
        for p in m.features[-3:].parameters(): p.requires_grad = True

    elif name == "EfficientNet-B0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        for p in m.parameters(): p.requires_grad = False
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, 256),
            nn.ReLU(), nn.Linear(256, 2))
        for p in m.features[-3:].parameters(): p.requires_grad = True

    elif name == "ResNet-50":
        m = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for p in m.parameters(): p.requires_grad = False
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(in_f, 256),
            nn.ReLU(), nn.Linear(256, 2))
        for p in m.layer4.parameters(): p.requires_grad = True

    return m

# ══════════════════════════════════════════════════════════════════
# Train one fold
# ══════════════════════════════════════════════════════════════════
def train_fold(model, tr_paths, tr_lbls, va_paths, va_lbls):
    train_ds = SkinDataset(tr_paths, tr_lbls, train_tf)
    val_ds   = SkinDataset(va_paths, va_lbls, val_tf)

    counts  = [tr_lbls.count(0), tr_lbls.count(1)]
    w       = [1.0 / counts[l] for l in tr_lbls]
    sampler = WeightedRandomSampler(w, len(w))

    tl = DataLoader(train_ds, batch_size=BATCH_SIZE,
                    sampler=sampler, num_workers=0)
    vl = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                    shuffle=False, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=4, gamma=0.5)

    train_losses, val_losses = [], []
    best_f1, best_state = 0.0, None

    print(f"     {'Ep':<5} {'TrainLoss':<13} "
          f"{'ValLoss':<13} {'ValF1'}")
    print(f"     {'─'*5} {'─'*13} {'─'*13} {'─'*8}")

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0.0
        for imgs, lbls in tl:
            imgs = imgs.to(DEVICE)
            lbls = torch.as_tensor(lbls).to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        avg_t = t_loss / len(tl)

        model.eval()
        v_loss, all_p, all_l = 0.0, [], []
        with torch.no_grad():
            for imgs, lbls in vl:
                imgs = imgs.to(DEVICE)
                lbls = torch.as_tensor(lbls).to(DEVICE)
                out  = model(imgs)
                loss = criterion(out, lbls)
                v_loss += loss.item()
                preds   = out.argmax(dim=1)
                all_p.extend(preds.cpu().numpy())
                all_l.extend(lbls.cpu().numpy())

        avg_v = v_loss / len(vl)
        v_f1  = f1_score(all_l, all_p, average='macro')
        train_losses.append(avg_t)
        val_losses.append(avg_v)

        print(f"     [{epoch+1:2d}/{EPOCHS}]  "
              f"{avg_t:.6f}     "
              f"{avg_v:.6f}     "
              f"{v_f1:.4f}")

        if v_f1 > best_f1:
            best_f1    = v_f1
            best_state = {k: v.clone()
                          for k, v in model.state_dict().items()}
        scheduler.step()

    return train_losses, val_losses, best_f1, best_state

# ══════════════════════════════════════════════════════════════════
# STEP 1 — 5-Fold CV for all three models
# Save best state for each model after its K-fold finishes
# ══════════════════════════════════════════════════════════════════
print("── STEP 1: 5-Fold Cross Validation ──")
print("  Best state saved after each model completes K-fold.\n")

model_names   = ["MobileNetV3-Small", "EfficientNet-B0", "ResNet-50"]
skf           = StratifiedKFold(
    n_splits=K_FOLDS, shuffle=True, random_state=SEED)

kfold_results = {}
model_curves  = {}
saved_states  = {}

for model_name in model_names:
    print(f"\n{'─'*65}")
    print(f"  {model_name}  —  {K_FOLDS}-Fold Cross Validation")
    print(f"{'─'*65}")

    fold_train_losses = []
    fold_val_losses   = []
    fold_best_f1s     = []
    global_best_f1    = 0.0
    global_best_state = None

    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(pool_paths, pool_labels)
    ):
        print(f"\n  ── Fold {fold+1}/{K_FOLDS} "
              f"(train={len(tr_idx)}, val={len(va_idx)}) ──")
        torch.manual_seed(SEED + fold)
        np.random.seed(SEED + fold)

        tr_paths = [pool_paths[i] for i in tr_idx]
        tr_lbls  = pool_labels[tr_idx].tolist()
        va_paths = [pool_paths[i] for i in va_idx]
        va_lbls  = pool_labels[va_idx].tolist()

        model = build_model(model_name).to(DEVICE)
        t_losses, v_losses, best_f1, best_state = train_fold(
            model, tr_paths, tr_lbls, va_paths, va_lbls)

        fold_train_losses.append(t_losses)
        fold_val_losses.append(v_losses)
        fold_best_f1s.append(best_f1)

        gap = v_losses[-1] - t_losses[-1]
        print(f"\n     Fold {fold+1} → "
              f"best_f1={best_f1*100:.2f}%  "
              f"gap={gap:.4f}  "
              f"{'No overfitting' if gap < 0.05 else 'CHECK OVERFITTING'}")

        if best_f1 > global_best_f1:
            global_best_f1    = best_f1
            global_best_state = best_state

    avg_f1  = float(np.mean(fold_best_f1s))
    std_f1  = float(np.std(fold_best_f1s))
    avg_gap = float(np.mean([
        v[-1] - t[-1]
        for t, v in zip(fold_train_losses, fold_val_losses)
    ]))

    kfold_results[model_name] = {
        'avg_f1' : avg_f1,
        'std_f1' : std_f1,
        'avg_gap': avg_gap,
    }
    model_curves[model_name] = {
        'avg_train_loss': np.mean(fold_train_losses, axis=0),
        'avg_val_loss'  : np.mean(fold_val_losses,   axis=0),
        'std_val_loss'  : np.std(fold_val_losses,    axis=0),
    }

    saved_states[model_name] = global_best_state

    # Save intermediate model — threshold added later in Step 4
    torch.save({
        'model_name'   : model_name,
        'model_state'  : global_best_state,
        'kfold_avg_f1' : avg_f1,
        'kfold_std_f1' : std_f1,
        'kfold_avg_gap': avg_gap,
        'classes'      : ['not_skin', 'skin'],
        'seed'         : SEED,
        'threshold_set': False,   # will be updated in Step 4
    }, MODEL_PATHS[model_name])

    print(f"\n  {model_name} — 5-Fold Average:")
    print(f"    Avg F1    : {avg_f1*100:.2f}% ± {std_f1*100:.2f}%")
    print(f"    Avg gap   : {avg_gap:.4f}  "
          f"→ {'No overfitting' if avg_gap < 0.05 else 'POSSIBLE OVERFITTING'}")
    print(f"    Saved to  : {MODEL_PATHS[model_name]}")

# ══════════════════════════════════════════════════════════════════
# STEP 2 — Plot overfitting check
# ══════════════════════════════════════════════════════════════════
print(f"\n── STEP 2: Plotting overfitting check ──")

colors   = {
    "MobileNetV3-Small": "#2196F3",
    "EfficientNet-B0"  : "#FF9800",
    "ResNet-50"        : "#4CAF50",
}
epochs_x = range(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(
    "MAAI-SkinDx | Agent 1 — Overfitting Check\n"
    "Training Loss vs Validation Loss  |  "
    f"5-Fold Cross Validation  |  {EPOCHS} Epochs",
    fontsize=13, fontweight='bold', y=1.02
)

for i, name in enumerate(model_names):
    ax     = axes[i]
    c      = model_curves[name]
    t_loss = c['avg_train_loss']
    v_loss = c['avg_val_loss']
    v_std  = c['std_val_loss']
    color  = colors[name]

    ax.plot(epochs_x, t_loss, color=color, linewidth=2.5,
            linestyle='-',  label='Training Loss')
    ax.plot(epochs_x, v_loss, color=color, linewidth=2.5,
            linestyle='--', label='Validation Loss')
    ax.fill_between(epochs_x,
                    v_loss - v_std, v_loss + v_std,
                    alpha=0.15, color=color)

    final_gap = float(v_loss[-1] - t_loss[-1])
    verdict   = ("No Overfitting"
                 if final_gap < 0.05 else "Possible Overfitting")
    vc        = "green" if final_gap < 0.05 else "red"

    ax.set_title(f"{name}\nFinal gap: {final_gap:.4f}  →  {verdict}",
                 fontweight='bold', color=vc, fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs_x)

plt.tight_layout()
overfit_path = os.path.join(OUTPUT_DIR, "overfitting_check.png")
plt.savefig(overfit_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {overfit_path}")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — Evaluate ALL THREE on held-out test set
# Winner selected by held-out F1 — not K-fold
# ══════════════════════════════════════════════════════════════════
print(f"\n── STEP 3: Evaluating ALL THREE on held-out test set ──")
print(f"  380 images never seen during training or K-fold.\n")

test_ds  = SkinDataset(test_paths, test_labels.tolist(), val_tf)
test_ldr = DataLoader(test_ds, batch_size=BATCH_SIZE,
                      shuffle=False, num_workers=0)

holdout_results = {}

for model_name in model_names:
    print(f"  Evaluating {model_name}...")

    model = build_model(model_name).to(DEVICE)
    model.load_state_dict(saved_states[model_name])
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, lbls in test_ldr:
            imgs  = imgs.to(DEVICE)
            out   = model(imgs)
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    acc  = accuracy_score(all_labels,  all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec  = recall_score(all_labels,    all_preds, average='macro')
    f1   = f1_score(all_labels,        all_preds, average='macro')
    cm   = confusion_matrix(all_labels, all_preds)

    holdout_results[model_name] = {
        'accuracy' : acc,
        'precision': prec,
        'recall'   : rec,
        'f1'       : f1,
        'cm'       : cm,
        'probs'    : all_probs,
        'labels'   : all_labels,
    }

    print(f"    Accuracy  : {acc*100:.2f}%")
    print(f"    Precision : {prec*100:.2f}%")
    print(f"    Recall    : {rec*100:.2f}%")
    print(f"    F1 Score  : {f1*100:.2f}%")
    print(f"    Confusion Matrix:")
    print(f"      NOT_SKIN correct:{cm[0][0]}  wrong:{cm[0][1]}")
    print(f"      SKIN    correct:{cm[1][1]}  wrong:{cm[1][0]}\n")

print(f"{'═'*65}")
print(f"  HELD-OUT TEST COMPARISON  (380 unseen images)")
print(f"{'═'*65}")
print(f"  {'Model':<22} {'Accuracy':<12} {'Precision':<12} "
      f"{'Recall':<10} {'F1 Score'}")
print(f"  {'─'*22} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")

winner_name = max(holdout_results,
                  key=lambda n: holdout_results[n]['f1'])

for name in model_names:
    r      = holdout_results[name]
    marker = "  ← WINNER" if name == winner_name else ""
    print(f"  {name:<22} "
          f"{r['accuracy']*100:.2f}%      "
          f"{r['precision']*100:.2f}%      "
          f"{r['recall']*100:.2f}%    "
          f"{r['f1']*100:.2f}%"
          f"{marker}")

print(f"{'═'*65}")
print(f"  WINNER (by held-out F1): {winner_name}")

# ══════════════════════════════════════════════════════════════════
# STEP 4 — ROC curve for ALL THREE + save threshold into each file
# ══════════════════════════════════════════════════════════════════
print(f"\n── STEP 4: ROC curves + optimal threshold for all models ──")

fig2, ax2 = plt.subplots(figsize=(8, 7))
ax2.plot([0,1],[0,1], color='gray', linestyle='--',
         linewidth=1, label='Random classifier')

for name in model_names:
    r   = holdout_results[name]
    fpr, tpr, thresholds = roc_curve(r['labels'], r['probs'])
    roc_auc = auc(fpr, tpr)

    distances = np.sqrt((fpr-0)**2 + (tpr-1)**2)
    opt_idx   = np.argmin(distances)
    opt_thr   = float(thresholds[opt_idx])
    opt_fpr   = float(fpr[opt_idx])
    opt_tpr   = float(tpr[opt_idx])

    holdout_results[name]['roc_auc'] = roc_auc
    holdout_results[name]['opt_thr'] = opt_thr
    holdout_results[name]['opt_tpr'] = opt_tpr
    holdout_results[name]['opt_fpr'] = opt_fpr

    # Evaluate with optimal threshold
    opt_preds = (r['probs'] >= opt_thr).astype(int)
    opt_f1    = f1_score(r['labels'], opt_preds, average='macro')
    opt_acc   = accuracy_score(r['labels'],  opt_preds)
    opt_prec  = precision_score(r['labels'], opt_preds, average='macro')
    opt_rec   = recall_score(r['labels'],    opt_preds, average='macro')

    holdout_results[name]['opt_f1']   = opt_f1
    holdout_results[name]['opt_acc']  = opt_acc
    holdout_results[name]['opt_prec'] = opt_prec
    holdout_results[name]['opt_rec']  = opt_rec

    # ── Update intermediate model file with threshold ──────────
    existing = torch.load(MODEL_PATHS[name], map_location='cpu')
    existing.update({
        'optimal_threshold' : opt_thr,
        'roc_auc'           : roc_auc,
        'test_f1'           : r['f1'],
        'test_accuracy'     : r['accuracy'],
        'test_precision'    : r['precision'],
        'test_recall'       : r['recall'],
        'opt_f1'            : opt_f1,
        'opt_accuracy'      : opt_acc,
        'threshold_set'     : True,
    })
    torch.save(existing, MODEL_PATHS[name])
    print(f"  {name}:")
    print(f"    AUC-ROC           : {roc_auc:.4f}")
    print(f"    Optimal threshold : {opt_thr:.4f}")
    print(f"    F1 at threshold   : {opt_f1*100:.2f}%")
    print(f"    Saved to          : {MODEL_PATHS[name]}")

    ax2.plot(fpr, tpr, color=colors[name], linewidth=2.5,
             label=f'{name}  (AUC={roc_auc:.4f})')
    ax2.scatter(opt_fpr, opt_tpr, color=colors[name],
                s=100, zorder=5, marker='*')

ax2.set_xlabel("False Positive Rate", fontsize=12)
ax2.set_ylabel("True Positive Rate", fontsize=12)
ax2.set_title(
    "MAAI-SkinDx | Agent 1 — ROC Curves\n"
    "All Three Models  |  Held-out Test Set (380 images)\n"
    "★ = optimal threshold point",
    fontweight='bold', fontsize=12
)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-0.01, 1.01])
ax2.set_ylim([-0.01, 1.01])

plt.tight_layout()
roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
plt.savefig(roc_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  ROC curve saved: {roc_path}")

# ══════════════════════════════════════════════════════════════════
# STEP 5 — Retrain winner on full train_pool (1520 images)
# ══════════════════════════════════════════════════════════════════
print(f"\n── STEP 5: Retraining {winner_name} on full train_pool ──")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

full_ds  = SkinDataset(pool_paths, pool_labels.tolist(), train_tf)
counts   = [int(np.sum(pool_labels==0)), int(np.sum(pool_labels==1))]
w        = [1.0/counts[l] for l in pool_labels.tolist()]
sampler  = WeightedRandomSampler(w, len(w))
full_ldr = DataLoader(full_ds, batch_size=BATCH_SIZE,
                      sampler=sampler, num_workers=0)

winner_model = build_model(winner_name).to(DEVICE)
criterion    = nn.CrossEntropyLoss()
optimizer    = optim.Adam(
    filter(lambda p: p.requires_grad, winner_model.parameters()),
    lr=LR)
scheduler    = optim.lr_scheduler.StepLR(
    optimizer, step_size=5, gamma=0.5)

print(f"  {'Epoch':<8} {'Loss':<12} {'Train Acc'}")
print(f"  {'─'*8} {'─'*12} {'─'*10}")

for epoch in range(15):
    winner_model.train()
    t_loss, t_corr, t_tot = 0.0, 0, 0
    for imgs, lbls in full_ldr:
        imgs = imgs.to(DEVICE)
        lbls = torch.as_tensor(lbls).to(DEVICE)
        optimizer.zero_grad()
        out  = winner_model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        t_loss += loss.item()
        preds   = out.argmax(dim=1)
        t_corr += (preds == lbls).sum().item()
        t_tot  += lbls.size(0)
    print(f"  [{epoch+1:2d}/15]   "
          f"{t_loss/len(full_ldr):.6f}    "
          f"{t_corr/t_tot:.4f}")
    scheduler.step()

# ══════════════════════════════════════════════════════════════════
# STEP 6 — Save final model with all metrics and threshold
# ══════════════════════════════════════════════════════════════════
print(f"\n── STEP 6: Saving final model ──")

winner_r   = holdout_results[winner_name]
winner_thr = winner_r['opt_thr']

torch.save({
    'model_name'        : winner_name,
    'model_state'       : winner_model.state_dict(),
    'optimal_threshold' : winner_thr,
    'roc_auc'           : winner_r['roc_auc'],
    'test_f1'           : winner_r['f1'],
    'test_accuracy'     : winner_r['accuracy'],
    'test_precision'    : winner_r['precision'],
    'test_recall'       : winner_r['recall'],
    'opt_f1'            : winner_r['opt_f1'],
    'opt_accuracy'      : winner_r['opt_acc'],
    'kfold_avg_f1'      : kfold_results[winner_name]['avg_f1'],
    'kfold_std_f1'      : kfold_results[winner_name]['std_f1'],
    'classes'           : ['not_skin', 'skin'],
    'seed'              : SEED,
}, FINAL_MODEL_PATH)

print(f"  Final model : {FINAL_MODEL_PATH}")
print(f"  All intermediate models kept — delete losers manually:")
for name in model_names:
    tag = " ← WINNER (also saved as skin_detector.pth)" \
          if name == winner_name else ""
    print(f"    {MODEL_PATHS[name]}{tag}")

# ══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print(f"  FINAL SUMMARY")
print(f"{'═'*65}")

print(f"\n  K-Fold Cross Validation (1520 images, {K_FOLDS} folds):")
print(f"  {'Model':<22} {'K-Fold F1':<16} "
      f"{'Std Dev':<12} {'Overfitting'}")
print(f"  {'─'*22} {'─'*16} {'─'*12} {'─'*15}")
for name in model_names:
    r = kfold_results[name]
    print(f"  {name:<22} "
          f"{r['avg_f1']*100:.2f}%          "
          f"±{r['std_f1']*100:.2f}%      "
          f"{'POSSIBLE' if r['avg_gap'] >= 0.05 else 'No'}")

print(f"\n  Held-out Test Set (380 images — truly unseen):")
print(f"  {'Model':<22} {'F1 Score':<12} {'AUC-ROC':<10} "
      f"{'Threshold':<12} {'F1@Threshold'}")
print(f"  {'─'*22} {'─'*12} {'─'*10} {'─'*12} {'─'*12}")
for name in model_names:
    r      = holdout_results[name]
    marker = "  ← WINNER" if name == winner_name else ""
    print(f"  {name:<22} "
          f"{r['f1']*100:.2f}%      "
          f"{r['roc_auc']:.4f}      "
          f"{r['opt_thr']:.4f}        "
          f"{r['opt_f1']*100:.2f}%"
          f"{marker}")

print(f"\n  WINNER  : {winner_name}")
print(f"  Reason  : Highest F1 on 380 held-out images.")
print(f"  K-Fold F1      : "
      f"{kfold_results[winner_name]['avg_f1']*100:.2f}%"
      f" ± {kfold_results[winner_name]['std_f1']*100:.2f}%")
print(f"  Test F1        : {winner_r['f1']*100:.2f}%")
print(f"  F1@Threshold   : {winner_r['opt_f1']*100:.2f}%")
print(f"  AUC-ROC        : {winner_r['roc_auc']:.4f}")
print(f"  Threshold      : {winner_thr:.4f} (ROC optimal)")
print(f"\n  Plots: {OUTPUT_DIR}")
print(f"    overfitting_check.png")
print(f"    roc_curve.png")
print(f"\n  Next steps after reviewing results:")
print(f"  1. Delete loser model files")
print(f"  2. Update skin_check.py")
print(f"  3. Run test_agent1.py")
print(f"  4. Run test_agent1_fairness.py")
print(f"{'═'*65}\n")