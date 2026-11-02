# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Evaluation — Step 17: Statistical Significance
# Bootstrap confidence intervals on F1 improvement.
# ══════════════════════════════════════════════════════════════════

import os, sys, json
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.eval_utils import (
    load_test_predictions, OOD_THRESHOLD, MALIGNANT_CLASSES, SEED)

np.random.seed(SEED)

N_BOOTSTRAP = 10000
ALPHA       = 0.05

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "evaluation", "statistical_significance.json")

CLASS_NAMES = [
    "Melanoma", "Melanocytic Nevus", "Basal Cell Carcinoma",
    "Actinic Keratosis / SCC", "Benign Keratosis",
    "Vascular Lesion", "Dermatofibroma", "Inflammatory",
    "Eczema / Dermatitis",
]

print("\n" + "=" * 65)
print("  MAAI-SkinDx | Step 17: Statistical Significance")
print(f"  Bootstrap samples: {N_BOOTSTRAP:,}   CI: {int((1-ALPHA)*100)}%")
print("=" * 65)

preds, confs, probs, labels, class_names = load_test_predictions()
in_dist   = confs >= OOD_THRESHOLD
n_total   = len(preds)
n_in_dist = int(in_dist.sum())
idx_all   = np.arange(n_total)
idx_ind   = np.where(in_dist)[0]

f1_s1_obs = f1_score(labels, preds,
    average="macro", zero_division=0) * 100
f1_s2_obs = f1_score(labels[in_dist], preds[in_dist],
    average="macro", zero_division=0) * 100
delta_obs = f1_s2_obs - f1_s1_obs

print(f"\n  Observed:")
print(f"    System 1 (Agent 3 alone)   : {f1_s1_obs:.4f}%")
print(f"    System 2 (+OOD detection)  : {f1_s2_obs:.4f}%")
print(f"    Improvement                : {delta_obs:+.4f}%")

print(f"\n  Running {N_BOOTSTRAP:,} bootstrap iterations ...")
f1_s1_b = np.zeros(N_BOOTSTRAP)
f1_s2_b = np.zeros(N_BOOTSTRAP)

for b in range(N_BOOTSTRAP):
    if (b+1) % 2000 == 0:
        print(f"    {b+1:,}/{N_BOOTSTRAP:,} ...")
    i1 = np.random.choice(idx_all, size=n_total, replace=True)
    f1_s1_b[b] = f1_score(labels[i1], preds[i1],
        average="macro", zero_division=0) * 100
    i2 = idx_ind[np.random.choice(
        len(idx_ind), size=len(idx_ind), replace=True)]
    f1_s2_b[b] = f1_score(labels[i2], preds[i2],
        average="macro", zero_division=0) * 100

delta_b     = f1_s2_b - f1_s1_b
s1_ci       = [float(np.percentile(f1_s1_b, 2.5)),
               float(np.percentile(f1_s1_b, 97.5))]
s2_ci       = [float(np.percentile(f1_s2_b, 2.5)),
               float(np.percentile(f1_s2_b, 97.5))]
delta_ci    = [float(np.percentile(delta_b, 2.5)),
               float(np.percentile(delta_b, 97.5))]
p_value     = float((delta_b <= 0).mean())
significant = p_value < ALPHA

print(f"\n  System 1:  {f1_s1_obs:.2f}%  95%CI [{s1_ci[0]:.2f}%, {s1_ci[1]:.2f}%]")
print(f"  System 2:  {f1_s2_obs:.2f}%  95%CI [{s2_ci[0]:.2f}%, {s2_ci[1]:.2f}%]")
print(f"  Delta:    {delta_obs:+.2f}%  95%CI [{delta_ci[0]:+.2f}%, {delta_ci[1]:+.2f}%]")
print(f"  p-value:   {p_value:.4f}  {'SIGNIFICANT ✓' if significant else 'NOT SIGNIFICANT ✗'}")

# Per-class results
print(f"\n  Computing per-class bootstrap CIs ...")
per_class = {}
for ci, cn in enumerate(CLASS_NAMES):
    b1 = np.zeros(N_BOOTSTRAP)
    b2 = np.zeros(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        i1 = np.random.choice(idx_all, size=n_total, replace=True)
        b1[b] = f1_score(labels[i1], preds[i1], average=None,
            zero_division=0, labels=list(range(9)))[ci] * 100
        i2 = idx_ind[np.random.choice(
            len(idx_ind), size=len(idx_ind), replace=True)]
        b2[b] = f1_score(labels[i2], preds[i2], average=None,
            zero_division=0, labels=list(range(9)))[ci] * 100
    obs1 = float(f1_score(labels, preds, average=None,
        zero_division=0, labels=list(range(9)))[ci] * 100)
    obs2 = float(f1_score(labels[in_dist], preds[in_dist],
        average=None, zero_division=0, labels=list(range(9)))[ci] * 100)
    per_class[cn] = {
        "obs_s1": round(obs1, 2), "obs_s2": round(obs2, 2),
        "delta": round(obs2-obs1, 2),
        "s1_95ci": [round(float(np.percentile(b1, 2.5)), 2),
                    round(float(np.percentile(b1, 97.5)), 2)],
        "s2_95ci": [round(float(np.percentile(b2, 2.5)), 2),
                    round(float(np.percentile(b2, 97.5)), 2)],
    }

print(f"\n  Per-class F1:")
print(f"  {'Class':<42} {'S1':>6}  {'S2':>6}  {'Delta':>7}")
print(f"  {'-'*60}")
for cn, r in per_class.items():
    print(f"  {cn:<42} {r['obs_s1']:>5.1f}%  "
          f"{r['obs_s2']:>5.1f}%  {r['delta']:>+6.1f}%")

output = {
    "date": datetime.now().isoformat(),
    "n_bootstrap": N_BOOTSTRAP, "alpha": ALPHA,
    "n_total": n_total, "n_in_distribution": n_in_dist,
    "ood_threshold": OOD_THRESHOLD,
    "system_1": {
        "label": "Agent 3 alone",
        "n_cases": n_total,
        "macro_f1_obs": round(f1_s1_obs, 4),
        "macro_f1_ci_95": [round(s1_ci[0], 4), round(s1_ci[1], 4)],
        "bootstrap_mean": round(float(f1_s1_b.mean()), 4),
        "bootstrap_std":  round(float(f1_s1_b.std()), 4),
    },
    "system_2": {
        "label": "Agent 3 + OOD Detection",
        "n_cases": n_in_dist,
        "macro_f1_obs": round(f1_s2_obs, 4),
        "macro_f1_ci_95": [round(s2_ci[0], 4), round(s2_ci[1], 4)],
        "bootstrap_mean": round(float(f1_s2_b.mean()), 4),
        "bootstrap_std":  round(float(f1_s2_b.std()), 4),
    },
    "improvement": {
        "delta_observed": round(delta_obs, 4),
        "delta_ci_95": [round(delta_ci[0], 4), round(delta_ci[1], 4)],
        "p_value": round(p_value, 6),
        "significant": significant,
        "conclusion": (
            f"The {delta_obs:.2f}% improvement is statistically "
            f"significant (p={p_value:.4f} < alpha={ALPHA}). "
            f"95% CI on improvement: [{delta_ci[0]:+.2f}%, {delta_ci[1]:+.2f}%]."
        ) if significant else (
            f"Not statistically significant (p={p_value:.4f})."
        ),
    },
    "per_class_results": per_class,
}
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {OUTPUT_PATH}")
print(f"{'='*65}\n")
