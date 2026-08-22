# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Orchestrator Test
# ══════════════════════════════════════════════════════════════════
# Tests the full pipeline end-to-end.
# Covers all agents including Clinical Agent with symptom data.
# ══════════════════════════════════════════════════════════════════

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from orchestrator.orchestrator import Orchestrator
from paths_config import (
    TEST_SKIN_IMAGE,
    TEST_NONSKIN_IMAGE,
    REAL_CLINICAL_IMAGE,
    REAL_DERMOSCOPIC_IMAGE,
)

orchestrator = Orchestrator()

# ── Test 1: Skin photo — no symptoms ──────────────────────────────
print(f"\n{'─' * 60}")
print(f"  Test: Skin photo — expect PASS through all agents")
result = orchestrator.process(TEST_SKIN_IMAGE)
print(f"\n  Final status  : {result['final_status']}")
print(f"  Pipeline      : {result['pipeline_status']}")
print(f"  Time          : {result['elapsed_seconds']}s")

# ── Test 2: Non-skin photo — should fail at Agent 1 ───────────────
print(f"\n{'─' * 60}")
print(f"  Test: Non-skin photo — expect FAIL at Agent 1")
result = orchestrator.process(TEST_NONSKIN_IMAGE)
print(f"\n  Final status  : {result['final_status']}")
print(f"  Pipeline      : {result['pipeline_status']}")
print(f"  Time          : {result['elapsed_seconds']}s")

# ── Test 3: Real clinical sample — no symptoms ────────────────────
print(f"\n{'─' * 60}")
print(f"  Test: Real clinical sample — expect PASS through all agents")
result = orchestrator.process(REAL_CLINICAL_IMAGE)
print(f"\n  Final status  : {result['final_status']}")
print(f"  Pipeline      : {result['pipeline_status']}")
print(f"  Time          : {result['elapsed_seconds']}s")

# ── Test 4: Real dermoscopic — no symptoms ────────────────────────
print(f"\n{'─' * 60}")
print(f"  Test: Real dermoscopic — expect PASS through all agents")
result = orchestrator.process(REAL_DERMOSCOPIC_IMAGE)
print(f"\n  Final status  : {result['final_status']}")
print(f"  Pipeline      : {result['pipeline_status']}")
print(f"  Time          : {result['elapsed_seconds']}s")

# ── Test 5: Skin photo WITH symptoms — HIGH risk patient ──────────
# This is the key test for the Clinical Agent.
# Patient profile: 67yo female, lesion changed, grew, elevated,
# itches, has skin cancer history, diameter 12mm.
# Expected: HIGH clinical risk, elevated ML probability.
print(f"\n{'─' * 60}")
print(f"  Test: Skin photo WITH symptoms — expect Clinical Agent PASS")

high_risk_symptoms = {
    "age"                : 67,
    "gender"             : "FEMALE",
    "fitspatrick"        : 3,
    "region"             : "FACE",
    "itch"               : "True",
    "grew"               : "True",
    "hurt"               : "False",
    "changed"            : "True",
    "bleed"              : "False",
    "elevation"          : "True",
    "skin_cancer_history": True,
    "cancer_history"     : False,
    "smoke"              : False,
    "drink"              : False,
    "diameter_1"         : 12.0,
    "diameter_2"         : 8.0,
}

result = orchestrator.process(
    image_path = TEST_SKIN_IMAGE,
    symptoms   = high_risk_symptoms
)
print(f"\n  Final status  : {result['final_status']}")
print(f"  Pipeline      : {result['pipeline_status']}")
print(f"  Time          : {result['elapsed_seconds']}s")

# ── Test 6: Skin photo WITH low risk symptoms ─────────────────────
# Patient profile: 28yo male, no symptoms, no history.
# Expected: LOW clinical risk, low ML probability.
print(f"\n{'─' * 60}")
print(f"  Test: Skin photo WITH low risk symptoms — expect LOW risk")

low_risk_symptoms = {
    "age"                : 28,
    "gender"             : "MALE",
    "fitspatrick"        : 2,
    "region"             : "ARM",
    "itch"               : "False",
    "grew"               : "False",
    "hurt"               : "False",
    "changed"            : "False",
    "bleed"              : "False",
    "elevation"          : "False",
    "skin_cancer_history": False,
    "cancer_history"     : False,
    "smoke"              : False,
    "drink"              : False,
    "diameter_1"         : 5.0,
    "diameter_2"         : 4.0,
}

result = orchestrator.process(
    image_path = TEST_SKIN_IMAGE,
    symptoms   = low_risk_symptoms
)
print(f"\n  Final status  : {result['final_status']}")
print(f"  Pipeline      : {result['pipeline_status']}")
print(f"  Time          : {result['elapsed_seconds']}s")