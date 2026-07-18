# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Central Paths Configuration
# ══════════════════════════════════════════════════════════════════
# One place for every file/folder path used across the project.
# Never type a full path by hand anywhere else — import it from here.
# Works on any machine/OS because PROJECT_ROOT is calculated, not typed.
# ══════════════════════════════════════════════════════════════════

import os

# This file's own folder = the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ─── Common folders ─────────────────────────────────────────────
DATA_RAW_DIR       = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR         = os.path.join(PROJECT_ROOT, "models")

# ─── Specific test files ────────────────────────────────────────
TEST_SKIN_IMAGE        = os.path.join(DATA_RAW_DIR, "skin photo.jpeg")
TEST_NONSKIN_IMAGE     = os.path.join(DATA_RAW_DIR, "non skin photo.jpeg")
REAL_CLINICAL_IMAGE    = os.path.join(DATA_RAW_DIR, "real_clinical_sample.png")
REAL_DERMOSCOPIC_IMAGE = os.path.join(DATA_RAW_DIR, "real_dermoscopic_sample.jpg")
