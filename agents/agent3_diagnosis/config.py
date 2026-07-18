import torch

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME  = "ResNet-50"
MODEL_PATH  = "models/agent3/agent3_resnet50_final.pth"
IN_FEATURES = 2048
NUM_CLASSES = 9
IMG_SIZE    = 224

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
    0: "Melanoma — malignant, requires urgent referral",
    1: "Melanocytic Nevus — common benign mole",
    2: "Basal Cell Carcinoma — slow-growing skin cancer",
    3: "Actinic Keratosis / Squamous Cell Carcinoma",
    4: "Benign Keratosis — non-cancerous skin growth",
    5: "Vascular Lesion — blood vessel abnormality",
    6: "Dermatofibroma — benign fibrous nodule",
    7: "Inflammatory Skin Condition",
    8: "Eczema / Dermatitis — inflammatory skin condition",
}

# Melanoma safety threshold — clinical safety measure.
# If melanoma probability exceeds this value, flag as melanoma
# even if another class has slightly higher probability.
# Improves recall from 80.3% toward the 85% clinical target.
MELANOMA_SAFETY_THRESHOLD = 0.35

TOP_N = 5

# Model performance metrics
CV_F1       = 61.43   # single dataset comparison phase
HOLDOUT_F1  = 74.57   # multi-dataset final training
HOLDOUT_ACC = 76.94
DATASETS    = ["HAM10000", "Fitzpatrick-17k",
               "PAD-UFES-20", "MassiveBalanced"]