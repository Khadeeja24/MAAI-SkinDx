# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 1 — Create Held-Out Test Set
# ══════════════════════════════════════════════════════════════════
# Reserves 20% of images as a truly held-out test set.
# These images are NEVER seen during training or cross-validation.
# They are used ONCE only — for the final honest evaluation.
#
# Must be run BEFORE any retraining.
# ══════════════════════════════════════════════════════════════════

import os
import shutil
import random
import numpy as np
from pathlib import Path

# ─── Reproducibility ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIN_DIR     = os.path.join(PROJECT_ROOT, "data", "raw",
               "skin_detector_training", "skin")
NOT_SKIN_DIR = os.path.join(PROJECT_ROOT, "data", "raw",
               "skin_detector_training", "not_skin", "selected")

# Held-out test set destination
TEST_DIR     = os.path.join(PROJECT_ROOT, "data", "raw",
               "skin_detector_training", "test")
TEST_SKIN    = os.path.join(TEST_DIR, "skin")
TEST_NOT     = os.path.join(TEST_DIR, "not_skin")

# Training pool destination (remaining 80%)
POOL_DIR     = os.path.join(PROJECT_ROOT, "data", "raw",
               "skin_detector_training", "train_pool")
POOL_SKIN    = os.path.join(POOL_DIR, "skin")
POOL_NOT     = os.path.join(POOL_DIR, "not_skin")

HOLDOUT_RATIO = 0.20

print(f"\n{'═' * 65}")
print(f"  MAAI-SkinDx | Creating Held-Out Test Set")
print(f"{'═' * 65}")
print(f"  Holdout ratio : {HOLDOUT_RATIO*100:.0f}%")
print(f"  Seed          : {SEED}")
print(f"{'═' * 65}\n")

# ── Collect skin images ────────────────────────────────────────────
skin_images = []
for subfolder in ['dermoscopic', 'clinical']:
    folder = os.path.join(SKIN_DIR, subfolder)
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                skin_images.append(os.path.join(folder, f))

# ── Collect not-skin images ────────────────────────────────────────
not_skin_images = []
for f in os.listdir(NOT_SKIN_DIR):
    if f.lower().endswith(('.jpg', '.jpeg', '.png')):
        not_skin_images.append(os.path.join(NOT_SKIN_DIR, f))

print(f"  Total skin images     : {len(skin_images)}")
print(f"  Total not-skin images : {len(not_skin_images)}")

# ── Stratified split — preserve class ratio ────────────────────────
random.shuffle(skin_images)
random.shuffle(not_skin_images)

n_skin_test     = int(len(skin_images)     * HOLDOUT_RATIO)
n_not_skin_test = int(len(not_skin_images) * HOLDOUT_RATIO)

skin_test     = skin_images[:n_skin_test]
skin_train    = skin_images[n_skin_test:]
not_skin_test = not_skin_images[:n_not_skin_test]
not_skin_train= not_skin_images[n_not_skin_test:]

print(f"\n  Test set  — skin: {len(skin_test)}   "
      f"not-skin: {len(not_skin_test)}   "
      f"total: {len(skin_test)+len(not_skin_test)}")
print(f"  Train pool— skin: {len(skin_train)}   "
      f"not-skin: {len(not_skin_train)}   "
      f"total: {len(skin_train)+len(not_skin_train)}")

# ── Create folders ─────────────────────────────────────────────────
for d in [TEST_SKIN, TEST_NOT, POOL_SKIN, POOL_NOT]:
    os.makedirs(d, exist_ok=True)

# ── Copy test set ──────────────────────────────────────────────────
print(f"\n  Copying held-out test set...")
for img in skin_test:
    shutil.copy2(img, os.path.join(TEST_SKIN, os.path.basename(img)))
for img in not_skin_test:
    shutil.copy2(img, os.path.join(TEST_NOT, os.path.basename(img)))

# ── Copy training pool ─────────────────────────────────────────────
print(f"  Copying training pool...")
for img in skin_train:
    shutil.copy2(img, os.path.join(POOL_SKIN, os.path.basename(img)))
for img in not_skin_train:
    shutil.copy2(img, os.path.join(POOL_NOT, os.path.basename(img)))

# ── Verify ─────────────────────────────────────────────────────────
test_skin_count  = len(list(Path(TEST_SKIN).glob("*.*")))
test_not_count   = len(list(Path(TEST_NOT).glob("*.*")))
pool_skin_count  = len(list(Path(POOL_SKIN).glob("*.*")))
pool_not_count   = len(list(Path(POOL_NOT).glob("*.*")))

print(f"\n{'═' * 65}")
print(f"  HELD-OUT TEST SET CREATED")
print(f"{'═' * 65}")
print(f"  Test set (LOCKED — use once only at the very end):")
print(f"    Skin     : {test_skin_count} images")
print(f"    Not-skin : {test_not_count} images")
print(f"    Total    : {test_skin_count + test_not_count} images")
print(f"    Path     : {TEST_DIR}")
print(f"\n  Training pool (for K-fold cross validation):")
print(f"    Skin     : {pool_skin_count} images")
print(f"    Not-skin : {pool_not_count} images")
print(f"    Total    : {pool_skin_count + pool_not_count} images")
print(f"    Path     : {POOL_DIR}")
print(f"\n  IMPORTANT:")
print(f"  Do NOT look at test set results until final evaluation.")
print(f"  K-fold must use ONLY the train_pool folder.")
print(f"{'═' * 65}\n")