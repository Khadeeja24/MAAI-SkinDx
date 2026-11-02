# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Evaluation — Step 4: Conflict Resolution Rate
# ══════════════════════════════════════════════════════════════════

import os, sys, json, random
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.eval_utils import (
    load_test_predictions, OOD_THRESHOLD, MALIGNANT_CLASSES, SEED)

random.seed(SEED)
np.random.seed(SEED)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "evaluation", "conflict_resolution.json")
ESCALATE_DECISIONS = {"ESCALATE", "REFER_URGENT", "REVIEW", "REFER"}

print("\n" + "=" * 65)
print("  MAAI-SkinDx | Step 4: Conflict Resolution Rate")
print("=" * 65)

preds, confs, probs, labels, class_names = load_test_predictions()
in_dist   = confs >= OOD_THRESHOLD
n_total   = len(preds)
n_in_dist = in_dist.sum()

def resolve(pred_idx, clinical_risk, ml_prob):
    image_mal   = pred_idx in MALIGNANT_CLASSES
    symptom_mal = clinical_risk in ("HIGH", "CRITICAL") or ml_prob > 0.60
    if not image_mal and symptom_mal:
        return "IMAGE_LOW_SYMPTOM_HIGH", "ESCALATE"
    elif image_mal and not symptom_mal:
        return "IMAGE_HIGH_SYMPTOM_LOW", "REVIEW"
    elif image_mal and symptom_mal:
        return "CONCORDANT", "REFER_URGENT"
    else:
        return "CONCORDANT", "ROUTINE"

print(f"\n  Scenario 1: Clinical agent correctly reflects true label ...")
s1 = dict(total=0, conflicts=0, ilsh=0, ihsl=0, concordant=0,
          correct=0, mal_total=0, ben_total=0,
          correctly_escalated=0, false_escalated=0)

for i in range(n_total):
    if not in_dist[i]:
        continue
    true_mal  = int(labels[i]) in MALIGNANT_CLASSES
    clin_risk = "HIGH" if true_mal else "LOW"
    ml_prob   = 0.85  if true_mal else 0.15
    ct, dec   = resolve(int(preds[i]), clin_risk, ml_prob)
    esc       = dec in ESCALATE_DECISIONS

    s1["total"] += 1
    if true_mal: s1["mal_total"] += 1
    else:        s1["ben_total"] += 1
    if ct != "CONCORDANT": s1["conflicts"] += 1
    if ct == "IMAGE_LOW_SYMPTOM_HIGH": s1["ilsh"] += 1
    elif ct == "IMAGE_HIGH_SYMPTOM_LOW": s1["ihsl"] += 1
    else: s1["concordant"] += 1
    if true_mal and esc:     s1["correctly_escalated"] += 1
    if not true_mal and esc: s1["false_escalated"] += 1
    if (true_mal and esc) or (not true_mal and not esc):
        s1["correct"] += 1

n              = s1["total"]
conflict_rate  = s1["conflicts"] / n * 100
resolution_acc = s1["correct"]   / n * 100
sensitivity    = s1["correctly_escalated"] / max(s1["mal_total"], 1) * 100
specificity    = (s1["ben_total"] - s1["false_escalated"]) / max(s1["ben_total"], 1) * 100

print(f"    Conflicts detected       : {s1['conflicts']} ({conflict_rate:.1f}%)")
print(f"      IMAGE_LOW_SYMPTOM_HIGH : {s1['ilsh']}")
print(f"      IMAGE_HIGH_SYMPTOM_LOW : {s1['ihsl']}")
print(f"    Resolution accuracy      : {resolution_acc:.1f}%")
print(f"    Sensitivity (malignant)  : {sensitivity:.1f}%")
print(f"    Specificity (benign)     : {specificity:.1f}%")

print(f"\n  Scenario 2: Clinical agent 50% accurate (noise stress test) ...")
s2 = dict(total=0, conflicts=0, correct=0,
          correctly_escalated=0, mal_total=0)

for i in range(n_total):
    if not in_dist[i]:
        continue
    true_mal = int(labels[i]) in MALIGNANT_CLASSES
    if random.random() < 0.5:
        clin_risk = "HIGH" if true_mal else "LOW"
        ml_prob   = 0.80  if true_mal else 0.20
    else:
        clin_risk = "LOW"  if true_mal else "HIGH"
        ml_prob   = 0.20  if true_mal else 0.80
    ct, dec = resolve(int(preds[i]), clin_risk, ml_prob)
    esc = dec in ESCALATE_DECISIONS
    s2["total"] += 1
    if true_mal: s2["mal_total"] += 1
    if ct != "CONCORDANT": s2["conflicts"] += 1
    if true_mal and esc: s2["correctly_escalated"] += 1
    if (true_mal and esc) or (not true_mal and not esc):
        s2["correct"] += 1

n2 = s2["total"]
print(f"    Resolution accuracy : {s2['correct']/n2*100:.1f}%")
print(f"    Sensitivity         : "
      f"{s2['correctly_escalated']/max(s2['mal_total'],1)*100:.1f}%")

output = {
    "date": datetime.now().isoformat(),
    "n_total": n_total,
    "n_in_distribution": int(n_in_dist),
    "ood_threshold": OOD_THRESHOLD,
    "scenario_1_informed_clinical": {
        "description": "Clinical agent correctly reflects true label",
        "n_cases": n,
        "conflict_rate_pct": round(conflict_rate, 2),
        "image_low_symptom_high": s1["ilsh"],
        "image_high_symptom_low": s1["ihsl"],
        "concordant": s1["concordant"],
        "resolution_accuracy_pct": round(resolution_acc, 2),
        "sensitivity_pct": round(sensitivity, 2),
        "specificity_pct": round(specificity, 2),
        "correctly_escalated": s1["correctly_escalated"],
        "malignant_total": s1["mal_total"],
    },
    "scenario_2_noisy_clinical": {
        "description": "Clinical agent 50% accurate (noise stress test)",
        "n_cases": n2,
        "conflict_rate_pct": round(s2["conflicts"]/n2*100, 2),
        "resolution_accuracy_pct": round(s2["correct"]/n2*100, 2),
        "sensitivity_pct": round(
            s2["correctly_escalated"]/max(s2["mal_total"],1)*100, 2),
    },
}
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*65}")
print(f"  CONFLICT RESOLUTION COMPLETE")
print(f"  Sensitivity   : {sensitivity:.1f}%    Specificity: {specificity:.1f}%")
print(f"  Resolution Acc: {resolution_acc:.1f}%")
print(f"  Saved: {OUTPUT_PATH}")
print(f"{'='*65}\n")
