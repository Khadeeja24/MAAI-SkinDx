# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 1 — Image Quality Agent
# Check 3: Skin Detection
# ══════════════════════════════════════════════════════════════════
# Confirms the submitted image contains human skin.
#
# Model Selection — 5-Fold Cross Validation + Held-out Test Set
# ──────────────────────────────────────────────────────────────
# Three models trained and compared:
#
#   Model               K-Fold F1      Test F1   AUC     FP  FN
#   MobileNetV3-Small   98.75%±0.57%   97.09%    0.9975   8   3
#   EfficientNet-B0     99.01%±0.55%   97.88%    0.9986   7   1
#   ResNet-50           99.08%±0.74%   97.89%    0.9984   5   3  ← WINNER
#
#   FP = non-skin images wrongly entering the pipeline
#   FN = real skin images wrongly rejected
#
#   Winner selection reason:
#   ResNet-50 has the lowest false positive count (5).
#   In a medical diagnostic system, a non-skin image entering
#   the pipeline produces a meaningless diagnosis that could
#   mislead a clinician. A missed skin image only asks the
#   patient to resubmit — far less harmful.
#   Clinical safety priority was used over raw F1 score.
#
# Training data:
#   500 dermoscopic  (HAM10000)
#   500 clinical     (Fitzpatrick-17k, balanced across skin tones)
#   800 textures     (DTD dataset)
#   100 phone photos (real-world non-skin close-ups)
#   Total: 1520 training pool + 380 held-out test set
#
# Threshold:
#   Set by ROC curve analysis on held-out test set.
#   Optimal point = closest to (0,1) on ROC curve.
#   Loaded automatically from checkpoint — never hardcoded.
#
# Replaces:
#   YCbCr color thresholding (failed on dark lesions)
#   HoughCircles detection   (never activated — removed)
#
# YCbCr kept as emergency fallback if model file is missing.
# ══════════════════════════════════════════════════════════════════

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# ─── Model path ───────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)
MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "skin_detector", "skin_detector.pth"
)

# ─── Global state ─────────────────────────────────────────────────
# Overwritten at startup by load_skin_model().
# Do NOT set these manually — values come from the checkpoint.
SKIN_THRESHOLD  = 0.50
SKIN_MODEL_NAME = "Unknown"

# ─── Image preprocessing ──────────────────────────────────────────
# Must exactly match the transforms used during training.
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ══════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════

def load_skin_model():
    """
    Load the winning skin detector model from checkpoint.

    Reads model architecture name, weights, and ROC-optimal
    threshold directly from the saved checkpoint file.
    No hardcoded values — everything comes from the file.

    Called once at module import time.
    Model stays in memory for all subsequent predictions.

    Returns:
        torch.nn.Module ready for inference, or None if file missing.
    """
    global SKIN_THRESHOLD, SKIN_MODEL_NAME

    if not os.path.exists(MODEL_PATH):
        print(f"  [Skin Check] WARNING: Model not found at {MODEL_PATH}")
        print(f"  [Skin Check] Falling back to YCbCr color detection")
        return None

    try:
        checkpoint = torch.load(
            MODEL_PATH,
            map_location='cpu',
            weights_only=False
        )

        model_name      = checkpoint.get('model_name', 'MobileNetV3-Small')
        SKIN_MODEL_NAME = model_name

        # ── Build correct architecture ─────────────────────────────
        # Architecture must match what was used during training.
        if model_name == 'ResNet-50':
            model    = models.resnet50(weights=None)
            in_f     = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_f, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )

        elif model_name == 'EfficientNet-B0':
            model            = models.efficientnet_b0(weights=None)
            in_f             = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Linear(in_f, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            )

        else:
            # MobileNetV3-Small — default fallback architecture
            model            = models.mobilenet_v3_small(weights=None)
            model.classifier = nn.Sequential(
                nn.Linear(576, 256),
                nn.Hardswish(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 2)
            )

        # ── Load trained weights ───────────────────────────────────
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        # ── Load ROC-optimal threshold ─────────────────────────────
        if 'optimal_threshold' in checkpoint:
            SKIN_THRESHOLD   = checkpoint['optimal_threshold']
            threshold_source = "ROC curve optimal"
        else:
            SKIN_THRESHOLD   = 0.50
            threshold_source = "default fallback"

        # ── Print loading summary ──────────────────────────────────
        kfold_f1 = checkpoint.get('kfold_avg_f1', 0)
        test_f1  = checkpoint.get('test_f1',       0)
        roc_auc  = checkpoint.get('roc_auc',       0)

        print(f"  [Skin Check] Model     : {model_name}")
        print(f"  [Skin Check] K-Fold F1 : {kfold_f1:.4f}")
        print(f"  [Skin Check] Test F1   : {test_f1:.4f}")
        print(f"  [Skin Check] AUC-ROC   : {roc_auc:.4f}")
        print(f"  [Skin Check] Threshold : "
              f"{SKIN_THRESHOLD:.4f} ({threshold_source})")

        return model

    except Exception as e:
        print(f"  [Skin Check] WARNING: Could not load model — {e}")
        print(f"  [Skin Check] Falling back to YCbCr color detection")
        return None


# ── Load once at module import — stays in memory ──────────────────
SKIN_MODEL = load_skin_model()


# ══════════════════════════════════════════════════════════════════
# CNN Skin Classification (primary method)
# ══════════════════════════════════════════════════════════════════

def check_skin_cnn(image) -> tuple:
    """
    Classify image as skin or not-skin using the winning CNN model.

    The model architecture is determined at load time from the
    checkpoint. The threshold is also loaded from the checkpoint.
    Both were set by the model comparison pipeline.

    Args:
        image (np.ndarray): Input BGR image from OpenCV.

    Returns:
        tuple:
            has_skin  (bool)  : True if skin detected
            skin_prob (float) : Model confidence 0.0 to 1.0
    """
    try:
        # Convert BGR to RGB and then to PIL for transforms
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        tensor    = TRANSFORM(pil_image).unsqueeze(0)

        with torch.no_grad():
            outputs   = SKIN_MODEL(tensor)
            probs     = torch.softmax(outputs, dim=1)
            skin_prob = probs[0][1].item()
            has_skin  = skin_prob >= SKIN_THRESHOLD

        status = "PASS" if has_skin else "FAIL — No Skin Detected"
        print(f"  [Skin Check]    {SKIN_MODEL_NAME} "
              f"confidence : {skin_prob:.4f}    {status}")

        return has_skin, skin_prob

    except Exception as e:
        print(f"  [Skin Check]    CNN error: {e}")
        print(f"  [Skin Check]    Falling back to YCbCr")
        return check_skin_ycbcr(image)


# ══════════════════════════════════════════════════════════════════
# YCbCr Color Thresholding (emergency fallback only)
# ══════════════════════════════════════════════════════════════════

def check_skin_ycbcr(image) -> tuple:
    """
    Emergency fallback skin detection using YCbCr color thresholding.

    Only used if the CNN model file is missing or fails to load.
    Not used in normal operation.

    Known limitations:
        - Fails on close-up dermoscopic images (dark lesion regions)
        - Incorrectly accepts skin-colored objects (wood, beige walls)

    Args:
        image (np.ndarray): Input BGR image from OpenCV.

    Returns:
        tuple:
            has_skin        (bool)  : True if skin color detected
            skin_percentage (float) : Percentage of skin-colored pixels
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0,   133,  77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    skin_mask       = cv2.inRange(ycrcb, lower, upper)
    total_pixels    = image.shape[0] * image.shape[1]
    skin_pixels     = np.count_nonzero(skin_mask)
    skin_percentage = round((skin_pixels / total_pixels) * 100, 2)
    has_skin        = (skin_pixels / total_pixels) >= 0.05

    status = "PASS" if has_skin else "FAIL — No Skin Detected"
    print(f"  [Skin Check]    YCbCr fallback : "
          f"{skin_percentage}%    {status}")

    return has_skin, skin_percentage


# ══════════════════════════════════════════════════════════════════
# Main Skin Check — called by Agent 1
# ══════════════════════════════════════════════════════════════════

def check_skin(image) -> tuple:
    """
    Run skin detection on input image.

    Primary method  : CNN classifier (ResNet-50 or winner model)
    Fallback method : YCbCr color thresholding

    The Orchestrator calls Agent 1 which calls this function.
    This function never calls any other agent.

    Args:
        image (np.ndarray): Input BGR image from OpenCV.

    Returns:
        tuple:
            has_skin (bool) : True if skin confirmed
            method   (str)  : Detection method used
            details  (dict) : Full detection metadata
    """
    if SKIN_MODEL is not None:
        has_skin, confidence = check_skin_cnn(image)
        return has_skin, "cnn", {
            "model"          : SKIN_MODEL_NAME,
            "is_dermoscopic" : False,
            "skin_confidence": confidence,
            "threshold"      : SKIN_THRESHOLD,
        }
    else:
        has_skin, skin_pct = check_skin_ycbcr(image)
        return has_skin, "ycbcr", {
            "model"          : "YCbCr fallback",
            "is_dermoscopic" : False,
            "skin_percentage": skin_pct,
        }