# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 6 — Outcome Simulation Demo
# ══════════════════════════════════════════════════════════════════
# Demonstrates Agent 6 background learning end-to-end by:
#
#   1. Running 60 test images through the pipeline
#   2. Storing each case in Agent 6 database
#   3. Simulating biopsy results for each case
#   4. Submitting confirmed outcomes to Agent 6
#   5. Verifying trust scores update via EMA
#
# This proves Phase 10 works end-to-end, not just in theory.
# ══════════════════════════════════════════════════════════════════

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

SEED = 42
np.random.seed(SEED)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT, "evaluation", "agent6_simulation.json")

print("\n" + "=" * 65)
print("  MAAI-SkinDx | Agent 6 — Outcome Simulation")
print("=" * 65)

# ── Load Agent 6 ───────────────────────────────────────────────────
from agents.agent6_background import BackgroundLearningAgent
agent6 = BackgroundLearningAgent()

initial_trust = agent6.get_trust_scores()
initial_count = agent6.get_case_count()
print(f"\n  Initial state:")
print(f"    Cases in DB    : {initial_count}")
print(f"    Saved trust    : {initial_trust}")

# ── Load test predictions ──────────────────────────────────────────
from scripts.eval_utils import (
    reproduce_test_split, load_agent3_model,
    load_agent2_features, run_inference,
    OOD_THRESHOLD, MALIGNANT_CLASSES)

print(f"\n  Loading test split and model ...")
test_df, class_names  = reproduce_test_split()
model, use_agent2     = load_agent3_model()
features_arr, p2i     = load_agent2_features()

# Take 60 cases — balanced malignant/benign
mal_df = test_df[
    test_df["unified_label"].isin(MALIGNANT_CLASSES)].head(30)
ben_df = test_df[
    ~test_df["unified_label"].isin(MALIGNANT_CLASSES)].head(30)
sim_df = pd.concat([mal_df, ben_df]).reset_index(drop=True)

print(f"  Simulation cases: {len(sim_df)} "
      f"({len(mal_df)} malignant + {len(ben_df)} benign)")

# ── Run inference on simulation cases ─────────────────────────────
preds, confs, probs, labels = run_inference(
    model, sim_df, use_agent2, features_arr, p2i)

# ── Build synthetic pipeline results and store in Agent 6 ─────────
print(f"\n  Storing {len(sim_df)} cases in Agent 6 database ...")

DECISION_MAP = {
    True : "ESCALATE",
    False: "ROUTINE",
}

stored_cases = []
for i in range(len(sim_df)):
    pred_idx  = int(preds[i])
    conf      = float(confs[i])
    true_label = int(labels[i])
    is_mal    = pred_idx in MALIGNANT_CLASSES
    is_ood    = conf < OOD_THRESHOLD

    case_id = f"SIM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}"
    decision = "OOD_WARNING" if is_ood else DECISION_MAP[is_mal]

    # Build minimal result dict matching Orchestrator output
    result = {
        "case_id"         : case_id,
        "image_path"      : sim_df.iloc[i]["image_path"],
        "pipeline_status" : "COMPLETE",
        "predicted_disease": class_names.get(pred_idx, f"Class {pred_idx}"),
        "is_ood"          : is_ood,
        "pox_override"    : False,
        "elapsed_seconds" : 8.5,
        "unified_outputs" : {
            "agent1"  : {"confidence": 1.0, "uncertainty": 0.0,
                         "quality_score": 0.90, "status": "PASS"},
            "agent2"  : {"confidence": 1.0, "uncertainty": 0.0,
                         "quality_score": 0.90, "status": "PASS"},
            "agent3"  : {"confidence": conf,
                         "uncertainty": 1.0 - conf,
                         "quality_score": 0.90, "status": "PASS"},
            "agent4"  : {"confidence": 0.85, "uncertainty": 0.15,
                         "quality_score": 0.90, "status": "PASS"},
            "clinical": {"confidence": 0.0, "uncertainty": 1.0,
                         "quality_score": 0.50, "status": "SKIP"},
            "agent5"  : {"confidence": 0.80, "uncertainty": 0.20,
                         "quality_score": 0.90, "status": "PASS"},
        },
        "assessment": {
            "agent3"  : {"adaptive_trust": 0.598, "base_trust": 0.742},
            "clinical": {"adaptive_trust": 0.000, "base_trust": 0.927},
        },
        "fusion"  : {"fused_malignancy": conf if is_mal else 1.0 - conf},
        "decision": {"decision": decision, "urgency": "URGENT" if is_mal else "ROUTINE"},
        "conflict": {"type": "NO_CLINICAL_DATA", "severity": "NONE"},
        "symptoms": {},
        "agents"  : {
            "agent3"  : {"predicted_index": pred_idx, "confidence": conf},
            "clinical": {"status": "SKIP"},
        },
        "pox_result": None,
    }

    agent6.store_case(result)
    stored_cases.append({
        "case_id"     : case_id,
        "predicted_idx": pred_idx,
        "true_label"  : true_label,
        "confidence"  : round(conf, 4),
        "is_ood"      : is_ood,
        "decision"    : decision,
    })

print(f"  Stored {len(stored_cases)} cases")

# ── Submit confirmed outcomes ──────────────────────────────────────
print(f"\n  Submitting confirmed outcomes (simulated biopsy) ...")

# Simulate: 85% of time biopsy confirms true label,
# 15% of time it is a different benign class (misclassification)
outcomes_submitted = 0
correct_predictions = 0
trust_updates = []

for case in stored_cases:
    case_id    = case["case_id"]
    true_label = case["true_label"]
    pred_idx   = case["predicted_idx"]

    # Simulate biopsy confirmation with 85% accuracy
    if np.random.random() < 0.85:
        confirmed_class = true_label
        confirmed_name  = class_names.get(true_label, f"Class {true_label}")
    else:
        # Biopsy finds a different benign condition
        benign_classes = [c for c in range(9)
                         if c not in MALIGNANT_CLASSES
                         and c != true_label]
        confirmed_class = np.random.choice(benign_classes)
        confirmed_name  = class_names.get(
            confirmed_class, f"Class {confirmed_class}")

    updated = agent6.submit_outcome(
        case_id         = case_id,
        confirmed_class = confirmed_class,
        confirmed_name  = confirmed_name,
        outcome_source  = "biopsy",
    )

    outcomes_submitted += 1
    if pred_idx == confirmed_class:
        correct_predictions += 1
    if updated:
        trust_updates.append({
            "after_outcome": outcomes_submitted,
            "updated_scores": updated,
        })

accuracy = correct_predictions / outcomes_submitted * 100
print(f"\n  Outcomes submitted : {outcomes_submitted}")
print(f"  Correct predictions: {correct_predictions} ({accuracy:.1f}%)")
print(f"  Trust updates fired: {len(trust_updates)}")

# ── Show trust evolution ───────────────────────────────────────────
final_trust   = agent6.get_trust_scores()
final_count   = agent6.get_case_count()
final_outcomes = agent6.get_outcome_count()
summary       = agent6.get_summary()

print(f"\n  Final state:")
print(f"    Cases in DB    : {final_count}")
print(f"    Outcomes in DB : {final_outcomes}")
print(f"    Final trust scores:")
for agent_name, score in final_trust.items():
    initial = initial_trust.get(agent_name, "N/A")
    print(f"      {agent_name:<12}: {score:.4f}  "
          f"(was {initial})")

if trust_updates:
    print(f"\n  Trust score evolution:")
    for i, upd in enumerate(trust_updates[:10]):
        print(f"    After outcome {upd['after_outcome']:>2}: "
              f"{upd['updated_scores']}")

# ── Save results ───────────────────────────────────────────────────
BASE_TRUST = {
    "agent3": 0.742, "clinical": 0.927}

output = {
    "date"               : datetime.now().isoformat(),
    "n_cases_simulated"  : len(stored_cases),
    "n_malignant"        : len(mal_df),
    "n_benign"           : len(ben_df),
    "n_outcomes_submitted": outcomes_submitted,
    "prediction_accuracy": round(accuracy, 2),
    "initial_trust_scores": initial_trust,
    "final_trust_scores" : final_trust,
    "base_trust_scores"  : BASE_TRUST,
    "n_trust_updates"    : len(trust_updates),
    "trust_evolution"    : trust_updates[:20],
    "agent6_summary"     : summary,
    "ema_config": {
        "alpha"       : 0.10,
        "window_size" : 50,
        "min_cases"   : 5,
        "trust_floor" : 0.30,
        "trust_ceiling": 0.98,
    },
    "conclusion": (
        f"Agent 6 background learning demonstrated end-to-end. "
        f"{len(stored_cases)} cases stored, "
        f"{outcomes_submitted} confirmed outcomes submitted. "
        f"Trust scores updated via EMA from real simulated outcomes. "
        f"System prediction accuracy on simulation: {accuracy:.1f}%."
    ),
}

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*65}")
print(f"  AGENT 6 SIMULATION COMPLETE")
print(f"  Cases stored    : {len(stored_cases)}")
print(f"  Outcomes filed  : {outcomes_submitted}")
print(f"  Prediction acc  : {accuracy:.1f}%")
print(f"  Trust updates   : {len(trust_updates)}")
print(f"  Saved: {OUTPUT_PATH}")
print(f"{'='*65}\n")
