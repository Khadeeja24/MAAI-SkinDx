# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Evaluation — Calibration (ECE)
# ══════════════════════════════════════════════════════════════════
# Expected Calibration Error measures whether the model's
# confidence scores are trustworthy.
#
# If the model says 80% confidence, it should be right ~80% of
# the time. If not, the model is miscalibrated — overconfident
# or underconfident.
#
# ECE formula:
#   ECE = sum over bins of (|bin| / N) × |accuracy(bin) - confidence(bin)|
#
# We compute:
#   - ECE for System 1 (all images)
#   - ECE for System 2 (OOD filtered)
#   - Reliability diagram (calibration plot)
#   - Max Calibration Error (MCE)
# ══════════════════════════════════════════════════════════════════

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.eval_utils import (
    load_test_predictions, OOD_THRESHOLD, MALIGNANT_CLASSES)

OUTPUT_JSON = os.path.join(
    PROJECT_ROOT, "evaluation", "calibration_ece.json")
OUTPUT_PLOT = os.path.join(
    PROJECT_ROOT, "evaluation", "calibration_plot.png")

N_BINS = 10

print("\n" + "=" * 65)
print("  MAAI-SkinDx | Calibration — Expected Calibration Error")
print("=" * 65)

# ── Load predictions ───────────────────────────────────────────────
preds, confs, probs, labels, class_names = load_test_predictions()

n_total = len(preds)
in_dist = confs >= OOD_THRESHOLD

# Correct predictions
correct_all = (preds == labels).astype(float)
correct_ind = correct_all[in_dist]
confs_ind   = confs[in_dist]

# ── ECE computation ────────────────────────────────────────────────

def compute_ece(confidences, correctness, n_bins=10):
    """Compute Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    mce  = 0.0
    n    = len(confidences)

    bin_accs   = []
    bin_confs  = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask   = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            bin_accs.append(0.0)
            bin_confs.append((lo + hi) / 2)
            bin_counts.append(0)
            continue
        bin_acc  = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_n    = mask.sum()
        gap      = abs(bin_acc - bin_conf)
        ece     += (bin_n / n) * gap
        mce      = max(mce, gap)
        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)
        bin_counts.append(int(bin_n))

    return ece, mce, bin_accs, bin_confs, bin_counts

# System 1 — all images
ece_s1, mce_s1, ba_s1, bc_s1, bn_s1 = compute_ece(
    confs, correct_all, N_BINS)

# System 2 — OOD filtered
ece_s2, mce_s2, ba_s2, bc_s2, bn_s2 = compute_ece(
    confs_ind, correct_ind, N_BINS)

print(f"\n  System 1 — Agent 3 alone:")
print(f"    ECE  : {ece_s1:.4f} ({ece_s1*100:.2f}%)")
print(f"    MCE  : {mce_s1:.4f} ({mce_s1*100:.2f}%)")
print(f"    Mean confidence : {confs.mean():.4f}")
print(f"    Mean accuracy   : {correct_all.mean():.4f}")
print(f"    Overconfident by: "
      f"{(confs.mean() - correct_all.mean())*100:+.2f}%")

print(f"\n  System 2 — +OOD detection:")
print(f"    ECE  : {ece_s2:.4f} ({ece_s2*100:.2f}%)")
print(f"    MCE  : {mce_s2:.4f} ({mce_s2*100:.2f}%)")
print(f"    Mean confidence : {confs_ind.mean():.4f}")
print(f"    Mean accuracy   : {correct_ind.mean():.4f}")
print(f"    Overconfident by: "
      f"{(confs_ind.mean() - correct_ind.mean())*100:+.2f}%")

# ── Reliability diagram ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
bins_mid  = np.linspace(0.05, 0.95, N_BINS)

for ax, ba, bc, bn, ece, label, color in [
    (axes[0], ba_s1, bc_s1, bn_s1, ece_s1,
     f"System 1 — Agent 3 alone (ECE={ece_s1:.3f})", "#2196F3"),
    (axes[1], ba_s2, bc_s2, bn_s2, ece_s2,
     f"System 2 — +OOD detection (ECE={ece_s2:.3f})", "#4CAF50"),
]:
    bar_width = 1.0 / N_BINS - 0.01
    ax.bar(bins_mid, ba, width=bar_width, alpha=0.7,
           color=color, label="Model accuracy per bin")
    ax.bar(bins_mid, bins_mid, width=bar_width, alpha=0.2,
           color="red", label="Gap (miscalibration)")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(label, fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.grid(alpha=0.3)

    # Bin counts as text
    for mid, acc, count in zip(bins_mid, ba, bn):
        if count > 0:
            ax.text(mid, min(acc + 0.03, 0.95),
                    str(count), ha="center", fontsize=7, alpha=0.7)

plt.suptitle(
    "MAAI-SkinDx | Reliability Diagram (Calibration)\n"
    "Insight Centre for Data Analytics | University of Galway",
    fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Calibration plot saved: {OUTPUT_PLOT}")

# ── Per-confidence-bin breakdown ───────────────────────────────────
print(f"\n  Calibration breakdown (System 2 — OOD filtered):")
print(f"  {'Bin':>12}  {'N':>6}  {'Conf':>6}  {'Acc':>6}  {'Gap':>7}")
print(f"  {'-'*44}")
bins_lo = np.linspace(0, 0.9, N_BINS)
bins_hi = np.linspace(0.1, 1.0, N_BINS)
for lo, hi, acc, conf, n in zip(
        bins_lo, bins_hi, ba_s2, bc_s2, bn_s2):
    if n > 0:
        gap = acc - conf
        print(f"  ({lo:.1f}, {hi:.1f}]  {n:>6}  "
              f"{conf:>6.3f}  {acc:>6.3f}  {gap:>+7.3f}")

# ── Save ───────────────────────────────────────────────────────────
output = {
    "date"         : datetime.now().isoformat(),
    "n_total"      : n_total,
    "n_in_dist"    : int(in_dist.sum()),
    "n_bins"       : N_BINS,
    "ood_threshold": OOD_THRESHOLD,
    "system_1": {
        "label"            : "Agent 3 alone",
        "ece"              : round(ece_s1, 4),
        "mce"              : round(mce_s1, 4),
        "mean_confidence"  : round(float(confs.mean()), 4),
        "mean_accuracy"    : round(float(correct_all.mean()), 4),
        "overconfidence"   : round(float(
            confs.mean() - correct_all.mean()), 4),
        "bin_accuracies"   : [round(x, 4) for x in ba_s1],
        "bin_confidences"  : [round(x, 4) for x in bc_s1],
        "bin_counts"       : bn_s1,
    },
    "system_2": {
        "label"            : "Agent 3 + OOD Detection",
        "ece"              : round(ece_s2, 4),
        "mce"              : round(mce_s2, 4),
        "mean_confidence"  : round(float(confs_ind.mean()), 4),
        "mean_accuracy"    : round(float(correct_ind.mean()), 4),
        "overconfidence"   : round(float(
            confs_ind.mean() - correct_ind.mean()), 4),
        "bin_accuracies"   : [round(x, 4) for x in ba_s2],
        "bin_confidences"  : [round(x, 4) for x in bc_s2],
        "bin_counts"       : bn_s2,
    },
    "interpretation": (
        f"System 1 ECE={ece_s1*100:.2f}%. "
        f"System 2 (OOD filtered) ECE={ece_s2*100:.2f}%. "
        f"OOD filtering {'improves' if ece_s2 < ece_s1 else 'changes'} "
        f"calibration by {abs(ece_s2-ece_s1)*100:.2f}%. "
        f"The model is "
        f"{'overconfident' if confs_ind.mean() > correct_ind.mean() else 'underconfident'} "
        f"on average."
    ),
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*65}")
print(f"  CALIBRATION EVALUATION COMPLETE")
print(f"  System 1 ECE : {ece_s1:.4f} ({ece_s1*100:.2f}%)")
print(f"  System 2 ECE : {ece_s2:.4f} ({ece_s2*100:.2f}%)")
print(f"  Saved: {OUTPUT_JSON}")
print(f"{'='*65}\n")
