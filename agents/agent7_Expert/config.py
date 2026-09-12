# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 7 Expert — Clinical Expert Agent Configuration
# ══════════════════════════════════════════════════════════════════

import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR      = os.path.join(
    PROJECT_ROOT, "outputs", "agent7_expert")
LLM_MODEL       = "openai/gpt-oss-120b"
LLM_MAX_TOKENS  = 1500
LLM_TEMPERATURE = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)

MALIGNANT_CLASSES = {0, 2, 3}

CLASS_FULL_NAMES = {
    0: "Melanoma",
    1: "Melanocytic Nevus (Common Mole)",
    2: "Basal Cell Carcinoma",
    3: "Actinic Keratosis / Squamous Cell Carcinoma",
    4: "Benign Keratosis (Seborrhoeic Keratosis)",
    5: "Vascular Lesion",
    6: "Dermatofibroma",
    7: "Inflammatory Skin Condition",
    8: "Eczema / Atopic Dermatitis",
}

URGENCY_LABELS = {
    "URGENT"          : "URGENT — Seek specialist assessment immediately",
    "WITHIN_1_WEEK"   : "HIGH PRIORITY — Specialist referral within 1 week",
    "WITHIN_4_WEEKS"  : "MODERATE — Clinical assessment within 4 weeks",
    "ROUTINE"         : "ROUTINE — Follow up in 6 to 12 months",
    "REFER_SPECIALIST": "REFER — Specialist consultation required",
}

FITZPATRICK_LABELS = {
    1: "Type I (Very Fair — Always burns, never tans)",
    2: "Type II (Fair — Usually burns, sometimes tans)",
    3: "Type III (Medium — Sometimes burns, always tans)",
    4: "Type IV (Olive — Rarely burns, always tans)",
    5: "Type V (Brown — Very rarely burns)",
    6: "Type VI (Dark — Never burns)",
}

REGION_LABELS = {
    "FACE"  : "Face",
    "NECK"  : "Neck",
    "CHEST" : "Chest",
    "BACK"  : "Back",
    "ARM"   : "Arm",
    "HAND"  : "Hand",
    "LEG"   : "Leg",
    "FOOT"  : "Foot",
    "SCALP" : "Scalp",
}