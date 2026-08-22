# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Clinical Agent — Model Training
# ══════════════════════════════════════════════════════════════════
# Trains a RandomForest classifier on PAD-UFES-20's 26 patient
# clinical features to predict malignancy probability.
#
# Target:
#   Malignant : MEL, BCC, SCC, ACK
#   Benign    : NEV, SEK
#
# Run once before using the Clinical Agent.
# Time: ~2 minutes on laptop.
# ══════════════════════════════════════════════════════════════════

import os
import sys
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, f1_score,
    roc_auc_score, accuracy_score
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agents.clinical_agent.config import (
    PAD_CSV, MODEL_PATH, MODEL_DIR,
    MALIGNANT_LABELS, ALL_FEATURES,
    REQUIRED_FEATURES, OPTIONAL_FEATURES
)

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"\n{'='*65}")
print(f"  MAAI-SkinDx | Clinical Agent — Model Training")
print(f"{'='*65}")

# ── Load PAD-UFES-20 ──────────────────────────────────────────────
print("\n── Loading PAD-UFES-20 ──")
df = pd.read_csv(PAD_CSV)
print(f"  Total rows : {len(df):,}")
print(f"  Columns    : {len(df.columns)}")

# ── Build binary target ───────────────────────────────────────────
df["malignant"] = df["diagnostic"].isin(MALIGNANT_LABELS).astype(int)
mal = df["malignant"].sum()
ben = len(df) - mal
print(f"\n  Malignant  : {mal:,} ({mal/len(df)*100:.1f}%)")
print(f"  Benign     : {ben:,} ({ben/len(df)*100:.1f}%)")

# ── Feature engineering ───────────────────────────────────────────
print("\n── Feature Engineering ──")

def encode_ternary(val):
    """
    Encode True/False/UNK values.
    True  = 1.0
    False = 0.0
    UNK   = 0.5  (uncertain — midpoint)
    NaN   = 0.5  (treat as unknown)
    """
    if pd.isna(val):
        return 0.5
    val = str(val).strip().lower()
    if val in ("true", "1", "yes"):
        return 1.0
    elif val in ("false", "0", "no"):
        return 0.0
    else:
        return 0.5  # UNK

def encode_bool(val):
    """Encode True/False/NaN. NaN → 0 (assume no)."""
    if pd.isna(val):
        return 0.0
    return 1.0 if str(val).strip().lower() in ("true", "1") else 0.0

def encode_gender(val):
    """FEMALE=0, MALE=1, NaN=0.5."""
    if pd.isna(val):
        return 0.5
    return 1.0 if str(val).strip().upper() == "MALE" else 0.0

# Symptom features (True/False/UNK)
for col in ["itch", "grew", "hurt", "changed", "bleed", "elevation"]:
    df[col] = df[col].apply(encode_ternary)

# Boolean features (True/False/NaN)
for col in ["skin_cancer_history", "cancer_history",
            "smoke", "drink"]:
    df[col] = df[col].apply(encode_bool)

# Gender
df["gender"] = df["gender"].apply(encode_gender)

# Fitzpatrick — NaN → median (3.0)
df["fitspatrick"] = pd.to_numeric(df["fitspatrick"], errors="coerce")
fitz_median = df["fitspatrick"].median()
df["fitspatrick"] = df["fitspatrick"].fillna(fitz_median)

# Diameter — NaN → median
d1_median = df["diameter_1"].median()
d2_median = df["diameter_2"].median()
df["diameter_1"] = df["diameter_1"].fillna(d1_median)
df["diameter_2"] = df["diameter_2"].fillna(d2_median)

# Age — already 0% NaN
df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(60)

print(f"  Features encoded: {len(ALL_FEATURES)}")
for f in ALL_FEATURES:
    null_count = df[f].isnull().sum()
    print(f"    {f:<25}: {null_count} NaN remaining")

# ── Build feature matrix ──────────────────────────────────────────
X = df[ALL_FEATURES].values.astype(np.float32)
y = df["malignant"].values

print(f"\n  Feature matrix : {X.shape}")
print(f"  Target         : {y.shape}")

# ── Train model ───────────────────────────────────────────────────
print("\n── Training RandomForest Classifier ──")
print("  Class weighting: balanced (handles 3.8:1 imbalance)")

model = RandomForestClassifier(
    n_estimators   = 300,
    max_depth      = 8,
    min_samples_split = 10,
    min_samples_leaf  = 5,
    class_weight   = "balanced",
    random_state   = 42,
    n_jobs         = -1,
)

# ── 5-fold cross-validation ───────────────────────────────────────
print("\n── 5-Fold Cross-Validation ──")
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scores  = cross_val_score(model, X, y, cv=skf,
                              scoring="f1", n_jobs=-1)
auc_scores = cross_val_score(model, X, y, cv=skf,
                              scoring="roc_auc", n_jobs=-1)
acc_scores = cross_val_score(model, X, y, cv=skf,
                              scoring="accuracy", n_jobs=-1)

print(f"  F1 Score : {f1_scores.mean()*100:.2f}% "
      f"(± {f1_scores.std()*100:.2f}%)")
print(f"  AUC-ROC  : {auc_scores.mean():.4f} "
      f"(± {auc_scores.std():.4f})")
print(f"  Accuracy : {acc_scores.mean()*100:.2f}% "
      f"(± {acc_scores.std()*100:.2f}%)")

# ── Train final model on all data ─────────────────────────────────
print("\n── Training Final Model on Full Dataset ──")
model.fit(X, y)

# Feature importance
importances = model.feature_importances_
feat_imp    = sorted(zip(ALL_FEATURES, importances),
                     key=lambda x: x[1], reverse=True)
print("\n  Feature Importances:")
for feat, imp in feat_imp:
    bar = "█" * int(imp * 100)
    print(f"    {feat:<25}: {imp:.4f}  {bar}")

# ── Save model ────────────────────────────────────────────────────
model_data = {
    "model"          : model,
    "features"       : ALL_FEATURES,
    "cv_f1_mean"     : float(f1_scores.mean()),
    "cv_f1_std"      : float(f1_scores.std()),
    "cv_auc_mean"    : float(auc_scores.mean()),
    "cv_acc_mean"    : float(acc_scores.mean()),
    "fitz_median"    : float(fitz_median),
    "d1_median"      : float(d1_median),
    "d2_median"      : float(d2_median),
    "malignant_labels": list(MALIGNANT_LABELS),
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_data, f)

print(f"\n{'='*65}")
print(f"  TRAINING COMPLETE")
print(f"{'='*65}")
print(f"  CV F1 Score : {f1_scores.mean()*100:.2f}%")
print(f"  CV AUC-ROC  : {auc_scores.mean():.4f}")
print(f"  Model saved : {MODEL_PATH}")
print(f"\n  Next: python notebooks/test_orchestrator.py")
print(f"{'='*65}\n")