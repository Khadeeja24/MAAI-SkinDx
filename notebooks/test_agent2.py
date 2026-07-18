# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 2 — Visual Feature Extraction Agent
# Test Script
# ══════════════════════════════════════════════════════════════════

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent2_feature_extraction import FeatureExtractionAgent
from paths_config import TEST_SKIN_IMAGE, REAL_CLINICAL_IMAGE, REAL_DERMOSCOPIC_IMAGE

# ─── Initialise Agent ──────────────────────────────────────────────
agent = FeatureExtractionAgent()

# ─── Test Cases ────────────────────────────────────────────────────
test_images = {
    "Personal skin photo"     : TEST_SKIN_IMAGE,
    "Real clinical sample"    : REAL_CLINICAL_IMAGE,
    "Real dermoscopic sample" : REAL_DERMOSCOPIC_IMAGE,
}

# ─── Run Tests ─────────────────────────────────────────────────────
for label, image_path in test_images.items():
    print(f"\n{'─' * 60}")
    print(f"  Test : {label}")
    print(f"{'─' * 60}")

    result = agent.run(image_path)

    print(f"\n── Final Result ──")
    print(f"Status         : {result['status']}")
    print(f"Reason         : {result['reason']}")

    if result['feature_vector'] is not None:
        print(f"Feature vector : shape={result['feature_vector'].shape}")
        print(f"ABCD scores    : {result['components']['abcd_scores']}")
        saved_path = agent.save_features(result['feature_vector'], image_path)
        loaded = np.load(saved_path)
        print(f"Reloaded shape : {loaded.shape}")
        print(f"Match original : {np.allclose(loaded, result['feature_vector'])}")