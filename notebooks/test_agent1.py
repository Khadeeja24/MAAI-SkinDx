# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 1 — Image Quality Agent
# Test Script
# ══════════════════════════════════════════════════════════════════

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent1_image_quality import ImageQualityAgent
from paths_config import TEST_SKIN_IMAGE, TEST_NONSKIN_IMAGE, REAL_CLINICAL_IMAGE, REAL_DERMOSCOPIC_IMAGE

# ─── Initialise Agent ──────────────────────────────────────────────
agent = ImageQualityAgent()

# ─── Test Cases ────────────────────────────────────────────────────
test_images = {
    "Skin photo — expect PASS"                                : TEST_SKIN_IMAGE,
    "Non-skin photo — expect FAIL"                             : TEST_NONSKIN_IMAGE,
    "Real clinical sample — expect PASS"                      : REAL_CLINICAL_IMAGE,
    "Real dermoscopic sample — expect PASS, dermoscope FOUND" : REAL_DERMOSCOPIC_IMAGE,
}

# ─── Run Tests ─────────────────────────────────────────────────────
for label, path in test_images.items():
    print(f"\n{'─' * 60}")
    print(f"  Test : {label}")
    result = agent.run(path)
    print(f"\n  ── Final Result ──")
    print(f"  Status : {result['status']}")
    print(f"  Reason : {result['reason']}")
    print(f"  Scores : {result['scores']}")