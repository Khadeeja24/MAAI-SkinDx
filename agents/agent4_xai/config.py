# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 4 — XAI Configuration
# ══════════════════════════════════════════════════════════════════

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
               os.path.abspath(__file__))))
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "agent4_xai")
IMG_SIZE     = 224
ALPHA        = 0.5    # heatmap overlay opacity (0=original, 1=heatmap)

# Written explanation templates per class and focus type
# focus_on_lesion  : model focused on lesion centre
# focus_on_border  : model focused on lesion border
# focus_diffuse    : model attention spread across image

CLASS_EXPLANATIONS = {
    0: {
        "focus_on_lesion": (
            "The model focused strongly on the lesion centre, detecting "
            "irregular pigmentation and asymmetric colour patterns that are "
            "hallmark features of melanoma."
        ),
        "focus_on_border": (
            "The model paid particular attention to the lesion border, "
            "detecting notching and irregularity characteristic of "
            "melanocytic malignancy."
        ),
        "focus_diffuse": (
            "The model assessed the overall lesion, capturing asymmetry "
            "and colour variation — key indicators under the ABCD criteria "
            "for melanoma."
        ),
    },
    1: {
        "focus_on_lesion": (
            "The model focused on the central pigmentation pattern, "
            "identifying regular symmetric features consistent with a "
            "benign melanocytic nevus (common mole)."
        ),
        "focus_on_border": (
            "The model examined the lesion border and found regular, "
            "well-defined edges consistent with a benign mole."
        ),
        "focus_diffuse": (
            "The model assessed the overall lesion appearance, finding "
            "uniform colour and symmetric shape consistent with a benign mole."
        ),
    },
    2: {
        "focus_on_lesion": (
            "The model focused on the lesion surface, detecting a translucent "
            "pearly texture characteristic of basal cell carcinoma — the most "
            "common skin cancer."
        ),
        "focus_on_border": (
            "The model detected rolled border patterns characteristic of "
            "nodular basal cell carcinoma."
        ),
        "focus_diffuse": (
            "The model captured surface texture and border patterns across "
            "the lesion consistent with basal cell carcinoma."
        ),
    },
    3: {
        "focus_on_lesion": (
            "The model focused on scaly, rough surface texture and irregular "
            "pigmentation consistent with actinic keratosis or squamous cell "
            "carcinoma — conditions requiring urgent assessment."
        ),
        "focus_on_border": (
            "The model detected irregular borders with surrounding skin changes "
            "consistent with actinic keratosis progressing to squamous cell "
            "carcinoma."
        ),
        "focus_diffuse": (
            "The model assessed widespread lesion characteristics including "
            "surface texture and colour changes consistent with actinic "
            "keratosis or squamous cell carcinoma."
        ),
    },
    4: {
        "focus_on_lesion": (
            "The model focused on the waxy, well-defined surface consistent "
            "with benign keratosis — a common non-cancerous skin growth."
        ),
        "focus_on_border": (
            "The model identified sharp, clearly defined borders "
            "characteristic of benign keratosis."
        ),
        "focus_diffuse": (
            "The model assessed the overall lesion appearance consistent "
            "with benign keratosis — a harmless skin growth."
        ),
    },
    5: {
        "focus_on_lesion": (
            "The model focused on vascular patterns and coloration consistent "
            "with a benign vascular lesion involving abnormal blood vessel "
            "formation."
        ),
        "focus_on_border": (
            "The model detected characteristic vascular border patterns "
            "consistent with a benign vascular abnormality."
        ),
        "focus_diffuse": (
            "The model captured the overall vascular appearance consistent "
            "with a benign blood vessel abnormality."
        ),
    },
    6: {
        "focus_on_lesion": (
            "The model focused on the firm raised surface texture consistent "
            "with dermatofibroma — a benign fibrous nodule common in skin."
        ),
        "focus_on_border": (
            "The model identified the characteristic dimpling sign at the "
            "lesion border consistent with dermatofibroma."
        ),
        "focus_diffuse": (
            "The model assessed overall lesion characteristics consistent "
            "with dermatofibroma — a benign fibrous skin nodule."
        ),
    },
    7: {
        "focus_on_lesion": (
            "The model focused on inflammatory patterns within the lesion, "
            "detecting features consistent with an inflammatory skin condition "
            "such as psoriasis or lichen planus."
        ),
        "focus_on_border": (
            "The model detected inflammatory border characteristics and "
            "surrounding skin involvement consistent with an inflammatory "
            "skin condition."
        ),
        "focus_diffuse": (
            "The model captured widespread inflammatory features consistent "
            "with an inflammatory skin condition."
        ),
    },
    8: {
        "focus_on_lesion": (
            "The model focused on the dry, inflamed surface consistent with "
            "eczema or dermatitis — a common inflammatory skin condition."
        ),
        "focus_on_border": (
            "The model detected ill-defined borders with surrounding skin "
            "involvement consistent with eczema or contact dermatitis."
        ),
        "focus_diffuse": (
            "The model assessed widespread skin involvement consistent with "
            "eczema or dermatitis."
        ),
    },
}

CLASS_URGENCY = {
    0: "URGENT — Malignant. Immediate clinical referral recommended.",
    1: "LOW — Benign. Routine monitoring advised.",
    2: "URGENT — Malignant. Clinical evaluation required.",
    3: "URGENT — Malignant or pre-malignant. Prompt assessment required.",
    4: "LOW — Benign. No immediate action required.",
    5: "LOW — Benign. Routine assessment advised.",
    6: "LOW — Benign. No immediate action required.",
    7: "MODERATE — Inflammatory. Clinical assessment recommended.",
    8: "MODERATE — Inflammatory. Treatment consultation recommended.",
}