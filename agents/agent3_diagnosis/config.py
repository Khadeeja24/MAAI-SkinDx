import torch

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/agent3/agent3_resnet50_final.pth"
NUM_CLASSES= 9
IMG_SIZE   = 224
TOP_N      = 5

CLASS_NAMES = {
    0: "Melanoma",
    1: "Melanocytic Nevus",
    2: "Basal Cell Carcinoma",
    3: "Actinic Keratosis / Squamous Cell Carcinoma",
    4: "Benign Keratosis",
    5: "Vascular Lesion",
    6: "Dermatofibroma",
    7: "Inflammatory",
    8: "Eczema / Dermatitis",
}

CLASS_FULL_NAMES = {
    0: "Melanoma — malignant, requires urgent clinical referral",
    1: "Melanocytic Nevus — common benign mole",
    2: "Basal Cell Carcinoma — slow-growing skin cancer",
    3: "Actinic Keratosis / Squamous Cell Carcinoma — pre-cancer or cancer",
    4: "Benign Keratosis — non-cancerous skin growth",
    5: "Vascular Lesion — blood vessel abnormality",
    6: "Dermatofibroma — benign fibrous nodule",
    7: "Inflammatory Skin Condition",
    8: "Eczema / Dermatitis — inflammatory skin condition",
}

# Malignant classes for binary evaluation and safety threshold
MALIGNANT_CLASSES         = {0, 2, 3}

# Clinical safety — if melanoma probability >= this value,
# flag as melanoma even if not the top predicted class.
# Improves melanoma recall toward the 85% clinical target.
MELANOMA_SAFETY_THRESHOLD = 0.20

# ── Model performance — Final Run (current model on disk) ──────────
# Architecture  : CombinedModel-ResNet50
# Input dim     : 8964 (backbone 2048 + Agent2 6916)
# Trained on    : 4 datasets, 144,872 images
# Best epoch    : 31 / 35
# Training time : 801.3 minutes
HOLDOUT_F1   = 74.18
HOLDOUT_ACC  = 76.93
BEST_EPOCH   = 31
ARCHITECTURE = "CombinedModel-ResNet50 (8964-dim)"
DATASETS     = ["HAM10000", "Fitzpatrick-17k",
                "PAD-UFES-20", "MassiveBalanced"]

# ── DDI-31 Benchmark Results ───────────────────────────────────────
# Paper: Daneshjou et al., Science Advances, 2022
# "Disparities in Dermatology AI Performance on a
#  Diverse, Curated Clinical Image Set"
# 656 images | 31 diseases | All 6 Fitzpatrick skin tones
# Pathologically confirmed | Never used in training
DDI31_IMAGES           = 656
DDI31_DISEASES         = 31
DDI31_BASELINE_SKINGPTX= 35.8    # SkinGPT-X published baseline
DDI31_ACCURACY         = 52.29   # MAAI-SkinDx — beats baseline
DDI31_IMPROVEMENT      = "+16.49 percentage points vs SkinGPT-X"
DDI31_MALIGNANT_RECALL = 73.10
DDI31_MALIGNANT_F1     = 44.50

# Fairness across Fitzpatrick skin tone groups
DDI31_FST_LIGHT_RECALL  = 81.63  # FST I-II  — acceptable
DDI31_FST_MEDIUM_RECALL = 85.14  # FST III-IV — acceptable
DDI31_FST_DARK_RECALL   = 45.83  # FST V-VI  — known limitation
# Note: dark skin gap due to insufficient dark skin malignant
# training data across all major public dermatology datasets.
# Planned for improvement in Phase 2 with additional data.