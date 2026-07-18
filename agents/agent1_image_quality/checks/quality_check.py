# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 1 — Image Quality Agent
# Check 2: Image Quality Assessment
# ══════════════════════════════════════════════════════════════════
# Evaluates overall perceptual image quality using BRISQUE.
# Catches noise, grain, compression artefacts, and exposure
# issues that Laplacian blur detection cannot detect.
#
# Method: BRISQUE (Blind/Referenceless Image Spatial Quality
#         Evaluator) via the piq library.
# Scale : 0–100. Lower score = higher quality.
# Threshold defined in config.py → BRISQUE_THRESHOLD
# ══════════════════════════════════════════════════════════════════

import cv2
import torch
import numpy as np
from PIL import Image
import piq
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BRISQUE_THRESHOLD


# ─── Quality Assessment ────────────────────────────────────────────

def check_quality(image) -> tuple:
    """
    Assess image quality using the BRISQUE no-reference metric.

    Args:
        image (np.ndarray): Input BGR image.

    Returns:
        tuple:
            is_poor_quality (bool)  : True if image fails quality threshold.
            score           (float) : BRISQUE score (0–100).
    """
    # OpenCV loads in BGR — convert to RGB for PIL and piq
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # piq expects a float32 tensor of shape [1, C, H, W] in range [0, 1]
    tensor = (
    torch.from_numpy(np.array(Image.fromarray(image_rgb)).astype(np.float32) / 255.0)
    .permute(2, 0, 1)
    .unsqueeze(0)
)

    score = round(piq.brisque(tensor, data_range=1.0).item(), 2)
    is_poor_quality = score > BRISQUE_THRESHOLD

    status = "FAIL — Poor Quality" if is_poor_quality else "PASS"
    print(f"  [Quality Check] Score    : {score:<10} Threshold : {BRISQUE_THRESHOLD:<10} {status}")

    return is_poor_quality, score