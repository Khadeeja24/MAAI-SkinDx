# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Orchestrator Test
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

test_cases = {
    "Skin photo — expect PASS through both agents"    : TEST_SKIN_IMAGE,
    "Non-skin photo — expect FAIL at Agent 1"         : TEST_NONSKIN_IMAGE,
    "Real clinical sample — expect PASS through both" : REAL_CLINICAL_IMAGE,
    "Real dermoscopic — expect PASS through both"     : REAL_DERMOSCOPIC_IMAGE,
}

for label, image_path in test_cases.items():
    print(f"\n{'─' * 60}")
    print(f"  Test: {label}")
    result = orchestrator.process(image_path)
    print(f"  Final status  : {result['final_status']}")
    print(f"  Pipeline      : {result['pipeline_status']}")
    print(f"  Time          : {result['elapsed_seconds']}s")