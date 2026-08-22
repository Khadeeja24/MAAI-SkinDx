# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Clinical Agent — Configuration
# ══════════════════════════════════════════════════════════════════

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
               os.path.abspath(__file__))))

PAD_CSV      = os.path.join(PROJECT_ROOT, "data", "raw",
               "pad_ufes20", "metadata.csv")
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models", "clinical_agent")
MODEL_PATH   = os.path.join(MODEL_DIR, "clinical_rf_model.pkl")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "clinical_agent")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Features used for ML model ─────────────────────────────────────
# Always available (0% NaN in PAD-UFES-20)
REQUIRED_FEATURES = [
    "age",
    "itch",
    "grew",
    "hurt",
    "changed",
    "bleed",
    "elevation",
]

# Optional features (35% NaN — imputed during training)
OPTIONAL_FEATURES = [
    "skin_cancer_history",
    "cancer_history",
    "fitspatrick",
    "gender",
    "diameter_1",
    "diameter_2",
    "smoke",
    "drink",
]

ALL_FEATURES = REQUIRED_FEATURES + OPTIONAL_FEATURES

# Malignant diagnoses — matches Agent 3 unified taxonomy
MALIGNANT_LABELS = {"MEL", "BCC", "SCC", "ACK"}
BENIGN_LABELS    = {"NEV", "SEK"}

# ── Rule-based risk weights ────────────────────────────────────────
# Evidence-based weights from dermatology clinical literature.
# Each factor adds points to a 0-100 risk score.

RISK_WEIGHTS = {
    # Lesion behaviour — strongest predictors
    "changed"              : 30,   # lesion changed recently
    "bleed"                : 25,   # lesion bleeds
    "grew"                 : 20,   # lesion grew
    "elevation"            : 10,   # lesion is raised
    "hurt"                 : 8,    # lesion hurts
    "itch"                 : 5,    # lesion itches

    # Patient history
    "skin_cancer_history"  : 20,   # previous skin cancer
    "cancer_history"       : 15,   # any previous cancer
    "smoke"                : 5,    # smoker
    "drink"                : 3,    # drinker

    # Age bands
    "age_over_70"          : 20,
    "age_60_to_70"         : 15,
    "age_50_to_60"         : 10,
    "age_under_50"         : 0,

    # Lesion size (diameter_1 in mm)
    "diameter_over_15"     : 15,
    "diameter_10_to_15"    : 8,
    "diameter_under_10"    : 0,

    # Skin tone (dark skin has higher clinical miss rate)
    "dark_skin"            : 5,    # FST V-VI
}

# Risk level thresholds
RISK_LEVELS = {
    "LOW"      : (0,  30),
    "MODERATE" : (30, 60),
    "HIGH"     : (60, 80),
    "CRITICAL" : (80, 101),
}

RISK_RECOMMENDATIONS = {
    "LOW"      : "Routine monitoring. Follow up in 6-12 months.",
    "MODERATE" : "Clinical assessment recommended within 4 weeks.",
    "HIGH"     : "Clinical assessment recommended within 1 week.",
    "CRITICAL" : "Urgent referral required. Seek assessment immediately.",
}