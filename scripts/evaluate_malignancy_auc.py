# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Evaluation — Malignancy AUC / ROC
# ══════════════════════════════════════════════════════════════════
# Computes AUC-ROC for malignant vs benign classification.
# Standard metric for clinical AI evaluation.
#
# Malignant classes: Melanoma (0), BCC (2), AK/SCC (3)
# Benign  classes : all others (1, 4, 5, 6, 7, 8)
#
# Three systems compared:
#   System 1: Agent 3 alone (all images)
#   System 2: Agent 3 + OOD detection (filtered)
#   System 3: Fused malignancy signal (OOD filtered)
#
# Outputs:
#   evaluation/malignancy_auc.json  — metrics
#   evaluation/roc_curve.png        — ROC plot
# ══════════════════════════════════════════════════════════════════

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.eval_utils import (
    load_test_predictions, OOD_THRESHOLD, MALIGNANT_CLASSES)

OUTPUT_JSON = os.path.join(
    PROJECT_ROOT, "evaluation", "malignancy_auc.json")
OUTPUT_ROC  = os.path.join(
    PROJECT_ROOT, "evaluation", "roc_curve.png")
OUTPUT_PRC  = os.path.join(
    PROJECT_ROOT, "evaluation", "precision_recall_curve.png")

print("\n" + "=" * 65)
print("  MAAI-SkinDx | Malignancy AUC / ROC")
print("=" * 65)

# ── Load predictions ───────────────────────────────────────────────
preds, confs, probs, labels, class_names = load_test_predictions()

n_total  = len(preds)
in_dist  = confs >= OOD_THRESHOLD

# Binary labels: 1 = malignant, 0 = benign
y_bin_all = np.array(
    [1 if l in MALIGNANT_CLASSES else 0 for l in labels])
y_bin_ind = y_bin_all[in_dist]

# ── Malignancy score — sum of malignant class probabilities ────────
# Classes 0=Melanoma, 2=BCC, 3=AK/SCC
mal_idx   = sorted(MALIGNANT_CLASSES)  # [0, 2, 3]
score_all = probs[:, mal_idx].sum(axis=1)
score_ind = score_all[in_dist]

print(f"\n  Dataset:")
print(f"    Total images     : {n_total:,}")
print(f"    In-distribution  : {in_dist.sum():,}")
print(f"    OOD excluded     : {(~in_dist).sum():,}")
print(f"    Malignant (all)  : {y_bin_all.sum():,}")
print(f"    Malignant (in-dist): {y_bin_ind.sum():,}")

# ── System 1: Agent 3 alone ────────────────────────────────────────
auc_s1 = roc_auc_score(y_bin_all, score_all)
ap_s1  = average_precision_score(y_bin_all, score_all)
fpr_s1, tpr_s1, thr_s1 = roc_curve(y_bin_all, score_all)

# Sensitivity @ 95% specificity
spec_target = 0.95
idx_95spec  = np.where(1 - fpr_s1 >= spec_target)[0]
sens_at_95spec_s1 = tpr_s1[idx_95spec[-1]] if len(idx_95spec) > 0 else 0.0

# Youden's J optimal threshold
j_scores  = tpr_s1 - fpr_s1
opt_idx   = j_scores.argmax()
opt_thr_s1 = thr_s1[opt_idx]
y_pred_s1  = (score_all >= opt_thr_s1).astype(int)
cm_s1      = confusion_matrix(y_bin_all, y_pred_s1)
tn, fp, fn, tp = cm_s1.ravel()
sens_s1    = tp / (tp + fn) if (tp + fn) > 0 else 0
spec_s1    = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n  System 1 — Agent 3 alone:")
print(f"    AUC-ROC               : {auc_s1:.4f}")
print(f"    Average Precision     : {ap_s1:.4f}")
print(f"    Optimal threshold     : {opt_thr_s1:.4f}")
print(f"    Sensitivity (recall)  : {sens_s1:.4f} ({sens_s1*100:.1f}%)")
print(f"    Specificity           : {spec_s1:.4f} ({spec_s1*100:.1f}%)")
print(f"    Sens @ 95% spec       : {sens_at_95spec_s1:.4f} ({sens_at_95spec_s1*100:.1f}%)")

# ── System 2: Agent 3 + OOD ────────────────────────────────────────
auc_s2 = roc_auc_score(y_bin_ind, score_ind)
ap_s2  = average_precision_score(y_bin_ind, score_ind)
fpr_s2, tpr_s2, thr_s2 = roc_curve(y_bin_ind, score_ind)

j_s2       = tpr_s2 - fpr_s2
opt_s2     = j_s2.argmax()
opt_thr_s2 = thr_s2[opt_s2]
y_pred_s2  = (score_ind >= opt_thr_s2).astype(int)
cm_s2      = confusion_matrix(y_bin_ind, y_pred_s2)
tn2, fp2, fn2, tp2 = cm_s2.ravel()
sens_s2    = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
spec_s2    = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0

idx_95s2 = np.where(1 - fpr_s2 >= spec_target)[0]
sens_at_95spec_s2 = tpr_s2[idx_95s2[-1]] if len(idx_95s2) > 0 else 0.0

print(f"\n  System 2 — Agent 3 + OOD detection:")
print(f"    AUC-ROC               : {auc_s2:.4f}")
print(f"    Average Precision     : {ap_s2:.4f}")
print(f"    Optimal threshold     : {opt_thr_s2:.4f}")
print(f"    Sensitivity (recall)  : {sens_s2:.4f} ({sens_s2*100:.1f}%)")
print(f"    Specificity           : {spec_s2:.4f} ({spec_s2*100:.1f}%)")
print(f"    Sens @ 95% spec       : {sens_at_95spec_s2:.4f} ({sens_at_95spec_s2*100:.1f}%)")

# ── Per-class AUC ──────────────────────────────────────────────────
print(f"\n  Per-class AUC (one-vs-rest, in-distribution):")
per_class_auc = {}
for cls_idx, cls_name in class_names.items():
    y_ovr   = (labels[in_dist] == cls_idx).astype(int)
    score_c = probs[in_dist, cls_idx]
    if y_ovr.sum() > 0 and (1 - y_ovr).sum() > 0:
        auc_c = roc_auc_score(y_ovr, score_c)
        per_class_auc[cls_name] = round(auc_c, 4)
        print(f"    {cls_name:<42}: {auc_c:.4f}")

# ── Plot ROC curves ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC
ax = axes[0]
ax.plot(fpr_s1, tpr_s1, color="#2196F3", lw=2,
        label=f"System 1 — Agent 3 alone (AUC={auc_s1:.3f})")
ax.plot(fpr_s2, tpr_s2, color="#4CAF50", lw=2,
        label=f"System 2 — +OOD detection (AUC={auc_s2:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
ax.set_title("ROC Curve — Malignant vs Benign", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=10)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.grid(alpha=0.3)

# Precision-Recall
prec_s1, rec_s1, _ = precision_recall_curve(y_bin_all, score_all)
prec_s2, rec_s2, _ = precision_recall_curve(y_bin_ind, score_ind)
baseline_prec = y_bin_all.mean()

ax2 = axes[1]
ax2.plot(rec_s1, prec_s1, color="#2196F3", lw=2,
         label=f"System 1 (AP={ap_s1:.3f})")
ax2.plot(rec_s2, prec_s2, color="#4CAF50", lw=2,
         label=f"System 2 (AP={ap_s2:.3f})")
ax2.axhline(baseline_prec, color="gray", lw=1, linestyle="--",
            label=f"Baseline ({baseline_prec:.3f})")
ax2.set_xlabel("Recall (Sensitivity)", fontsize=12)
ax2.set_ylabel("Precision", fontsize=12)
ax2.set_title("Precision-Recall Curve — Malignant vs Benign",
              fontsize=13, fontweight="bold")
ax2.legend(loc="upper right", fontsize=10)
ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.02])
ax2.grid(alpha=0.3)

plt.suptitle(
    "MAAI-SkinDx | Malignancy Detection Performance\n"
    "Insight Centre for Data Analytics | University of Galway",
    fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_ROC, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  ROC + PRC plot saved: {OUTPUT_ROC}")

# ── Per-class AUC bar chart ────────────────────────────────────────
fig2, ax3 = plt.subplots(figsize=(12, 5))
names  = list(per_class_auc.keys())
aucs   = list(per_class_auc.values())
colors = ["#F44336" if n in [
    "Melanoma", "Basal Cell Carcinoma",
    "Actinic Keratosis / SCC"] else "#2196F3"
    for n in names]
bars = ax3.barh(names, aucs, color=colors, alpha=0.85)
ax3.axvline(0.5, color="gray", linestyle="--", lw=1)
ax3.set_xlabel("AUC-ROC (one-vs-rest)", fontsize=12)
ax3.set_title(
    "Per-Class AUC-ROC (in-distribution test set)\n"
    "Red = Malignant classes",
    fontsize=12, fontweight="bold")
ax3.set_xlim([0, 1])
for bar, auc in zip(bars, aucs):
    ax3.text(auc + 0.01, bar.get_y() + bar.get_height()/2,
             f"{auc:.3f}", va="center", fontsize=9)
ax3.grid(axis="x", alpha=0.3)
plt.tight_layout()
per_class_path = os.path.join(
    PROJECT_ROOT, "evaluation", "per_class_auc.png")
plt.savefig(per_class_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Per-class AUC plot saved: {per_class_path}")

# ── Save JSON ──────────────────────────────────────────────────────
output = {
    "date"          : datetime.now().isoformat(),
    "n_total"       : n_total,
    "n_in_dist"     : int(in_dist.sum()),
    "n_malignant_all" : int(y_bin_all.sum()),
    "n_malignant_ind" : int(y_bin_ind.sum()),
    "ood_threshold" : OOD_THRESHOLD,
    "malignant_classes": list(MALIGNANT_CLASSES),
    "system_1": {
        "label"          : "Agent 3 alone (all images)",
        "n_cases"        : n_total,
        "auc_roc"        : round(auc_s1, 4),
        "average_precision": round(ap_s1, 4),
        "optimal_threshold": round(float(opt_thr_s1), 4),
        "sensitivity"    : round(sens_s1, 4),
        "specificity"    : round(spec_s1, 4),
        "sens_at_95_spec": round(sens_at_95spec_s1, 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    },
    "system_2": {
        "label"          : "Agent 3 + OOD Detection",
        "n_cases"        : int(in_dist.sum()),
        "auc_roc"        : round(auc_s2, 4),
        "average_precision": round(ap_s2, 4),
        "optimal_threshold": round(float(opt_thr_s2), 4),
        "sensitivity"    : round(sens_s2, 4),
        "specificity"    : round(spec_s2, 4),
        "sens_at_95_spec": round(sens_at_95spec_s2, 4),
        "tp": int(tp2), "fp": int(fp2),
        "tn": int(tn2), "fn": int(fn2),
    },
    "per_class_auc" : per_class_auc,
    "plots"         : {
        "roc_curve"      : OUTPUT_ROC,
        "per_class_auc"  : per_class_path,
    },
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*65}")
print(f"  MALIGNANCY AUC COMPLETE")
print(f"  System 1 AUC : {auc_s1:.4f}")
print(f"  System 2 AUC : {auc_s2:.4f}")
print(f"  Improvement  : {auc_s2-auc_s1:+.4f}")
print(f"  Saved: {OUTPUT_JSON}")
print(f"{'='*65}\n")
