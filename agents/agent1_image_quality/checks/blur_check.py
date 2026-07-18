# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 1 — Image Quality Agent
# Check 1: Blur Detection
# ══════════════════════════════════════════════════════════════════
# Detects whether an input image is too blurry for reliable
# diagnosis using Laplacian variance.
#
# Method: Laplacian filter (OpenCV)
# A sharp image produces high-variance edge responses.
# A blurry image produces low-variance, weak edge responses.
# Threshold defined in config.py → BLUR_THRESHOLD
# ══════════════════════════════════════════════════════════════════

import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BLUR_THRESHOLD


# ─── Blur Detection ────────────────────────────────────────────────

def check_blur(image) -> tuple:
    """
    Evaluate image sharpness using Laplacian variance.

    Args:
        image (np.ndarray): Input BGR image.

    Returns:
        tuple:
            is_blurry (bool)   : True if image fails blur threshold.
            variance  (float)  : Laplacian variance score.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = float(round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2))
    is_blurry = variance < BLUR_THRESHOLD

    status = "FAIL — Too Blurry" if is_blurry else "PASS"
    print(f"  [Blur Check]    Variance : {variance:<10} Threshold : {BLUR_THRESHOLD:<10} {status}")

    return is_blurry, variance