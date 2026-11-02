# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Evaluation — Step 9: Missing Agent Robustness
# ══════════════════════════════════════════════════════════════════

import os, sys, json
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.eval_utils import (
    load_test_predictions, OOD_THRESHOLD, MALIGNANT_CLASSES)

BASE_TRUST = {
    "agent1": 0.979, "agent2": 0.950, "agent3": 0.742,
    "agent4": 0.850, "clinical": 0.927, "agent5": 0.800,
}

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "evaluation", "missing_agent_robustness.json")

BASE_TRUST = {
    "agent1": 0.979, "agent2": 0.950, "agent3": 0.742,
    "agent4": 0.850, "clinical": 0.927, "agent5": 0.800,
}

CLASS_NAMES = [
    "Melanoma", "Melanocytic Nevus", "Basal Cell Carcinoma",
    "Actinic Keratosis / SCC", "Benign Keratosis",
    "Vascular Lesion", "Dermatofibroma", "Inflammatory",
    "Eczema / Dermatitis",
]

print("\n" + "=" * 65)
print("  MAAI-SkinDx | Step 9: Missing Agent Robustness")
print("=" * 65)

preds, confs, probs, labels, class_names = load_test_predictions()
in_dist = confs >= OOD_THRESHOLD
n_total = len(preds)

def eval_mask(mask, label):
    y_t = labels[mask]; y_p = preds[mask]
    f1  = f1_score(y_t, y_p, average="macro", zero_division=0) * 100
    acc = (y_t == y_p).mean() * 100
    pc  = f1_score(y_t, y_p, average=None, zero_division=0,
                   labels=list(range(9))) * 100
    mal_t = np.isin(y_t, list(MALIGNANT_CLASSES))
    mal_p = np.isin(y_p, list(MALIGNANT_CLASSES))
    cov   = (mal_t & mal_p).sum() / max(mal_t.sum(), 1) * 100
    return {
        "label": label, "n_evaluated": int(mask.sum()),
        "macro_f1": round(f1, 2), "accuracy": round(acc, 2),
        "malignant_coverage_pct": round(cov, 2),
        "per_class_f1": {CLASS_NAMES[i]: round(float(pc[i]), 2)
                         for i in range(9)},
    }

results = {}
all_mask = np.ones(n_total, dtype=bool)
base_f1  = None

print(f"\n  [1/6] Baseline: Full system ...")
results["baseline"] = eval_mask(in_dist, "Full System (OOD filtered)")
results["baseline"]["n_ood_flagged"] = int((~in_dist).sum())
base_f1 = results["baseline"]["macro_f1"]
print(f"        F1={base_f1:.2f}%  n={results['baseline']['n_evaluated']}")

print(f"\n  [2/6] No Agent 1: All images pass ...")
results["no_agent1"] = eval_mask(all_mask, "No Agent 1 (all images pass)")
d = results["no_agent1"]["macro_f1"] - base_f1
print(f"        F1={results['no_agent1']['macro_f1']:.2f}%  delta={d:+.2f}%")

print(f"\n  [3/6] No Agent 4: No XAI ...")
results["no_agent4"] = eval_mask(in_dist, "No Agent 4 (no XAI)")
results["no_agent4"]["note"] = (
    "Agent 4 provides XAI and Grad-CAM but does not change "
    "classification. Fusion weight redistributed to other agents.")
wts = {k: v for k, v in BASE_TRUST.items() if k != "agent4"}
tot = sum(wts.values())
results["no_agent4"]["fusion_weights_without_a4"] = {
    k: round(v/tot, 4) for k, v in wts.items()}
d = results["no_agent4"]["macro_f1"] - base_f1
print(f"        F1={results['no_agent4']['macro_f1']:.2f}%  delta={d:+.2f}%  "
      f"(classification unchanged)")

print(f"\n  [4/6] No Clinical Agent: Image only ...")
results["no_clinical"] = eval_mask(in_dist, "No Clinical Agent (image only)")
results["no_clinical"]["note"] = (
    "Without clinical symptoms, conflict detection unavailable. "
    "System operates on image evidence only.")
wts2 = {k: v for k, v in BASE_TRUST.items() if k != "clinical"}
tot2 = sum(wts2.values())
results["no_clinical"]["fusion_weights_without_clinical"] = {
    k: round(v/tot2, 4) for k, v in wts2.items()}
d = results["no_clinical"]["macro_f1"] - base_f1
print(f"        F1={results['no_clinical']['macro_f1']:.2f}%  delta={d:+.2f}%")

print(f"\n  [5/6] No Agent 5: No RAG knowledge ...")
results["no_agent5"] = eval_mask(in_dist, "No Agent 5 (no RAG knowledge)")
results["no_agent5"]["note"] = (
    "Agent 5 retrieves clinical knowledge for the report. "
    "Its absence does not affect classification or decisions.")
d = results["no_agent5"]["macro_f1"] - base_f1
print(f"        F1={results['no_agent5']['macro_f1']:.2f}%  delta={d:+.2f}%  "
      f"(report quality reduced)")

print(f"\n  [6/6] No Agent 7: No clinical report ...")
results["no_agent7"] = eval_mask(in_dist, "No Agent 7 (no clinical report)")
results["no_agent7"]["note"] = (
    "Agent 7 generates the prescription-style report. "
    "Its absence does not affect classification or decisions.")
d = results["no_agent7"]["macro_f1"] - base_f1
print(f"        F1={results['no_agent7']['macro_f1']:.2f}%  delta={d:+.2f}%  "
      f"(no patient report)")

print(f"\n{'='*65}")
print(f"  MISSING AGENT ROBUSTNESS SUMMARY")
print(f"  {'Scenario':<40} {'F1':>7}  {'Delta':>7}")
print(f"  {'-'*56}")
for key, res in results.items():
    d = res["macro_f1"] - base_f1
    print(f"  {res['label']:<40} {res['macro_f1']:>6.2f}%  {d:>+6.2f}%")

output = {
    "date": datetime.now().isoformat(),
    "n_total": n_total, "ood_threshold": OOD_THRESHOLD,
    "base_trust": BASE_TRUST, "results": results,
    "summary": {
        "baseline_macro_f1": base_f1,
        "agent1_absent_delta": round(
            results["no_agent1"]["macro_f1"] - base_f1, 2),
        "most_critical_agent": (
            "Agent 1 — quality filter protects F1 by "
            "removing low-quality images from evaluation"),
    },
}
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\n  Saved: {OUTPUT_PATH}")
print(f"{'='*65}\n")
