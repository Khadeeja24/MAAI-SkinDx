# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Pox Classifier — MLP Training
# Input  : 772-dim features (DINOv2 768 + ABCD 4)
# Output : 5-class pox classifier
# Classes: 0=Monkeypox, 1=Chickenpox, 2=Measles, 3=Cowpox, 4=HFMD
# ══════════════════════════════════════════════════════════════════

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ── Paths ─────────────────────────────────────────────────────────
FEAT_PATH  = os.path.join(
    PROJECT_ROOT, "data", "processed", "pox_features.npy")
LABEL_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "pox_labels.npy")
MODEL_DIR  = os.path.join(PROJECT_ROOT, "models", "pox_mlp")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "pox_mlp.pth")

# ── Config ────────────────────────────────────────────────────────
DEVICE     = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
SEED       = 42
TEST_SIZE  = 0.20
VAL_SIZE   = 0.10
BATCH_SIZE = 128
EPOCHS     = 100
LR         = 1e-3
PATIENCE   = 15

POX_CLASSES = {
    0: "Monkeypox",
    1: "Chickenpox",
    2: "Measles",
    3: "Cowpox",
    4: "HFMD",
}

torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"\n{'='*60}")
print(f"  MAAI-SkinDx | Pox MLP Training")
print(f"  Device  : {DEVICE}")
print(f"  Epochs  : {EPOCHS} (early stop patience {PATIENCE})")
print(f"{'='*60}\n")

# ── Load features ─────────────────────────────────────────────────
print("── Loading features ──")
X = np.load(FEAT_PATH).astype(np.float32)
y = np.load(LABEL_PATH).astype(np.int64)
print(f"  X shape : {X.shape}")
print(f"  y shape : {y.shape}")
for idx, name in POX_CLASSES.items():
    n = int((y == idx).sum())
    print(f"  Class {idx} {name:<15}: {n:,}")

# ── Split: train / val / test ──────────────────────────────────────
print(f"\n── Splitting data ──")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=TEST_SIZE,
    random_state=SEED, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=VAL_SIZE / (1 - TEST_SIZE),
    random_state=SEED, stratify=y_temp)

print(f"  Train : {len(X_train):,}")
print(f"  Val   : {len(X_val):,}")
print(f"  Test  : {len(X_test):,}")

# ── DataLoaders ───────────────────────────────────────────────────
def make_loader(X, y, shuffle=True):
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y))
    return DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=shuffle)

train_loader = make_loader(X_train, y_train, shuffle=True)
val_loader   = make_loader(X_val,   y_val,   shuffle=False)
test_loader  = make_loader(X_test,  y_test,  shuffle=False)

# ── Model ─────────────────────────────────────────────────────────
class PoxMLP(nn.Module):
    def __init__(self, input_dim=772, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)

model     = PoxMLP().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5, verbose=True)

total_params = sum(
    p.numel() for p in model.parameters())
print(f"\n  Model params: {total_params:,}")

# ── Training ──────────────────────────────────────────────────────
print(f"\n── Training ──")
best_val_f1   = 0.0
patience_count= 0
best_epoch    = 0
t0 = time.time()

for epoch in range(1, EPOCHS + 1):
    # Train
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        out  = model(X_b)
        loss = criterion(out, y_b)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(X_b)
    train_loss /= len(X_train)

    # Validate
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            out = model(X_b.to(DEVICE))
            val_preds.extend(
                out.argmax(1).cpu().numpy())
            val_true.extend(y_b.numpy())

    val_f1 = f1_score(
        val_true, val_preds, average="macro")
    scheduler.step(1 - val_f1)

    if epoch % 10 == 0 or epoch == 1:
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Elapsed: {elapsed:.0f}s")

    # Early stopping
    if val_f1 > best_val_f1:
        best_val_f1    = val_f1
        best_epoch     = epoch
        patience_count = 0
        torch.save({
            "epoch"      : epoch,
            "model_state": model.state_dict(),
            "val_f1"     : val_f1,
            "classes"    : POX_CLASSES,
            "input_dim"  : 772,
            "num_classes": 5,
        }, MODEL_PATH)
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print(f"\n  Early stop at epoch {epoch} "
                  f"(best epoch {best_epoch})")
            break

print(f"\n  Best val F1 : {best_val_f1:.4f} "
      f"at epoch {best_epoch}")
print(f"  Model saved : {MODEL_PATH}")

# ── Test evaluation ───────────────────────────────────────────────
print(f"\n── Test Evaluation ──")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

test_preds, test_true, test_probs = [], [], []
with torch.no_grad():
    for X_b, y_b in test_loader:
        out   = model(X_b.to(DEVICE))
        probs = torch.softmax(out, dim=1)
        test_preds.extend(out.argmax(1).cpu().numpy())
        test_true.extend(y_b.numpy())
        test_probs.extend(probs.cpu().numpy())

test_f1  = f1_score(test_true, test_preds, average="macro")
test_acc = (np.array(test_preds) ==
            np.array(test_true)).mean() * 100

print(f"\n  Test Accuracy : {test_acc:.2f}%")
print(f"  Test Macro F1 : {test_f1:.4f}")
print(f"\n  Per-class report:")
print(classification_report(
    test_true, test_preds,
    target_names=list(POX_CLASSES.values()),
    digits=4))

print(f"\n  Confusion Matrix:")
cm = confusion_matrix(test_true, test_preds)
header = "".join(
    [f"{POX_CLASSES[i][:6]:>8}" for i in range(5)])
print(f"{'':>12}{header}")
for i, row in enumerate(cm):
    row_str = "".join([f"{v:>8}" for v in row])
    print(f"  {POX_CLASSES[i][:10]:<12}{row_str}")

print(f"\n{'='*60}")
print(f"  COMPLETE")
print(f"  Model : {MODEL_PATH}")
print(f"  Test Macro F1 : {test_f1:.4f}")
print(f"\n  Next: integrate pox MLP into Orchestrator")
print(f"{'='*60}\n")