# ==================================================================
# MAAI-SkinDx | Confusion Matrix
# ==================================================================
# Uses the same eval_utils as all working eval scripts.
#
# Generates:
#   evaluation/confusion_matrix.png  — side-by-side normalised CMs
#   evaluation/confusion_matrix.json — raw counts + per-class F1
# ==================================================================

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.eval_utils import (
    load_test_predictions, OOD_THRESHOLD, MALIGNANT_CLASSES)

# ── Config ────────────────────────────────────────────────────────
CLASS_NAMES_FULL = [
    "Melanoma", "Melanocytic Nevus", "Basal Cell Carcinoma",
    "Actinic Keratosis / SCC", "Benign Keratosis",
    "Vascular Lesion", "Dermatofibroma", "Inflammatory",
    "Eczema / Dermatitis",
]
CLASS_NAMES_SHORT = [
    "MEL", "NV", "BCC", "AK/SCC",
    "BKL", "VASC", "DF", "INF", "ECZ",
]

OUTPUT_PNG  = os.path.join(PROJECT_ROOT, "evaluation", "confusion_matrix.png")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "evaluation", "confusion_matrix.json")

print("\n" + "=" * 65)
print("  MAAI-SkinDx | Confusion Matrix")
print("=" * 65)

# ── Load predictions ──────────────────────────────────────────────
print("\n  Loading test predictions ...")
preds, confs, probs, labels, class_names = load_test_predictions()
all_labels = np.array(labels)
all_preds  = np.array(preds)
all_confs  = np.array(confs)

n_total  = len(all_labels)
mask_ood = all_confs >= OOD_THRESHOLD
n_indist = int(mask_ood.sum())
n_ood    = n_total - n_indist

print(f"  Test set     : {n_total:,} images  9 classes")
print(f"  In-dist      : {n_indist:,}")
print(f"  OOD excluded : {n_ood:,}")

# ── System 1: Agent 3 alone (all images) ─────────────────────────
cm_s1     = confusion_matrix(all_labels, all_preds, labels=list(range(9)))
f1_s1     = f1_score(all_labels, all_preds, average="macro", zero_division=0)
acc_s1    = accuracy_score(all_labels, all_preds)
f1_cls_s1 = f1_score(all_labels, all_preds, average=None,
                     labels=list(range(9)), zero_division=0)

# ── System 2: In-distribution only ───────────────────────────────
y_true_s2 = all_labels[mask_ood]
y_pred_s2 = all_preds[mask_ood]
cm_s2     = confusion_matrix(y_true_s2, y_pred_s2, labels=list(range(9)))
f1_s2     = f1_score(y_true_s2, y_pred_s2, average="macro", zero_division=0)
acc_s2    = accuracy_score(y_true_s2, y_pred_s2)
f1_cls_s2 = f1_score(y_true_s2, y_pred_s2, average=None,
                     labels=list(range(9)), zero_division=0)

print(f"\n  System 1 — Agent 3 alone : Macro F1 {f1_s1*100:.2f}%  Acc {acc_s1*100:.2f}%")
print(f"  System 2 — In-dist only  : Macro F1 {f1_s2*100:.2f}%  Acc {acc_s2*100:.2f}%")


# ── Plot helper ───────────────────────────────────────────────────
def plot_cm(ax, cm, title, n, f1, acc, cmap):
    row_sum = cm.sum(axis=1, keepdims=True).astype(float)
    row_sum[row_sum == 0] = 1
    cm_norm = cm / row_sum

    im = ax.imshow(cm_norm, interpolation="nearest",
                   cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels(CLASS_NAMES_SHORT, fontsize=8,
                       rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES_SHORT, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=9, labelpad=6)
    ax.set_ylabel("True",      fontsize=9, labelpad=6)
    ax.set_title(
        f"{title}\n"
        f"N={n:,}   Macro F1={f1*100:.2f}%   Acc={acc*100:.2f}%",
        fontsize=9, pad=8
    )
    thresh = 0.5
    for i in range(9):
        for j in range(9):
            col = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i,
                    f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                    ha="center", va="center",
                    fontsize=6.5, color=col)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalised)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)


# ── Figure ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

plot_cm(axes[0], cm_s1,
        "System 1 — Agent 3 Alone\n(All test images, OOD included)",
        n_total, f1_s1, acc_s1, "Blues")

plot_cm(axes[1], cm_s2,
        "System 2 — Full Coordination System\n(In-distribution only, OOD excluded)",
        n_indist, f1_s2, acc_s2, "Greens")

fig.suptitle(
    "MAAI-SkinDx — Confusion Matrix\n"
    "Insight Centre for Data Analytics | University of Galway  |  "
    "Supervisor: Dr. Saeed Alsamhi",
    fontsize=11, y=1.01
)
fig.text(
    0.5, -0.01,
    "MEL=Melanoma  NV=Melanocytic Nevus  BCC=Basal Cell Carcinoma  "
    "AK/SCC=Actinic Keratosis/SCC  BKL=Benign Keratosis  "
    "VASC=Vascular Lesion  DF=Dermatofibroma  INF=Inflammatory  "
    "ECZ=Eczema/Dermatitis",
    ha="center", fontsize=7.5, style="italic", color="#555"
)

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)
fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Plot saved : {OUTPUT_PNG}")

# ── JSON ──────────────────────────────────────────────────────────
output = {
    "date"   : datetime.now().isoformat(),
    "system1": {
        "label"           : "Agent 3 alone (all images)",
        "n_total"         : int(n_total),
        "macro_f1"        : round(float(f1_s1) * 100, 2),
        "accuracy"        : round(float(acc_s1) * 100, 2),
        "confusion_matrix": cm_s1.tolist(),
        "per_class_f1"    : {
            CLASS_NAMES_FULL[i]: round(float(f1_cls_s1[i]) * 100, 2)
            for i in range(9)
        },
    },
    "system2": {
        "label"           : "Full coordination (in-distribution)",
        "n_total"         : int(n_indist),
        "n_ood_removed"   : n_ood,
        "macro_f1"        : round(float(f1_s2) * 100, 2),
        "accuracy"        : round(float(acc_s2) * 100, 2),
        "confusion_matrix": cm_s2.tolist(),
        "per_class_f1"    : {
            CLASS_NAMES_FULL[i]: round(float(f1_cls_s2[i]) * 100, 2)
            for i in range(9)
        },
    },
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)
print(f"  JSON saved : {OUTPUT_JSON}")

# ── Console summary ───────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"  CONFUSION MATRIX COMPLETE")
print(f"{'=' * 65}")
print(f"  {'Class':<32} {'S1 F1':>7}  {'S2 F1':>7}")
print(f"  {'-'*48}")
for i, name in enumerate(CLASS_NAMES_FULL):
    flag = "  ← malignant" if i in MALIGNANT_CLASSES else ""
    print(f"  {name:<32} {f1_cls_s1[i]*100:>6.2f}%  "
          f"{f1_cls_s2[i]*100:>6.2f}%{flag}")
print(f"  {'-'*48}")
print(f"  {'MACRO':<32} {f1_s1*100:>6.2f}%  {f1_s2*100:>6.2f}%")
print(f"\n  Saved: confusion_matrix.png + confusion_matrix.json")
print(f"{'=' * 65}\n")
