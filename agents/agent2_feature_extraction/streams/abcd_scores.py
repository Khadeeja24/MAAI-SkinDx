# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 2 — Visual Feature Extraction Agent
# Stream: ABCD Scores
# ══════════════════════════════════════════════════════════════════
# Computes the 4 standard dermoscopy clinical features mathematically
# using OpenCV. No model needed — pure image processing.
#
# A — Asymmetry   : How symmetric the lesion shape is
# B — Border      : How irregular the lesion border is
# C — Colour      : How many distinct colours are present
# D — Diameter    : Estimated relative size of the lesion
#
# These 4 values are appended to the PanDerm and Derm Foundation
# embeddings before passing the combined vector to Agent 3.
# ══════════════════════════════════════════════════════════════════

import cv2
import numpy as np


# ─── A: Asymmetry ──────────────────────────────────────────────────

def compute_asymmetry(mask: np.ndarray) -> float:
    """
    Measure lesion asymmetry by comparing top/bottom and left/right halves.

    Args:
        mask (np.ndarray): Binary mask of the lesion (255 = lesion, 0 = background).

    Returns:
        float: Asymmetry score. Higher = more asymmetric = more concerning.
    """
    h, w = mask.shape
    if h == 0 or w == 0:
        return 0.0

    # Split horizontally and vertically
    top    = mask[:h//2, :]
    bottom = mask[h//2:, :]
    left   = mask[:, :w//2]
    right  = mask[:, w//2:]

    # Flip to align halves
    bottom_flipped = cv2.flip(bottom, 0)
    right_flipped  = cv2.flip(right, 1)

    # Resize to match in case of odd dimensions
    top    = cv2.resize(top,    (w, h//2))
    bottom_flipped = cv2.resize(bottom_flipped, (w, h//2))
    left   = cv2.resize(left,   (w//2, h))
    right_flipped  = cv2.resize(right_flipped,  (w//2, h))

    # Asymmetry = difference between halves normalised by total lesion area
    total = np.sum(mask > 0) + 1e-6
    asym_h = np.sum(np.abs(top.astype(float) - bottom_flipped.astype(float))) / (255.0 * total)
    asym_v = np.sum(np.abs(left.astype(float) - right_flipped.astype(float)))  / (255.0 * total)

    return round((asym_h + asym_v) / 2.0, 4)


# ─── B: Border Irregularity ────────────────────────────────────────

def compute_border(mask: np.ndarray) -> float:
    """
    Measure border irregularity using contour compactness.
    A perfect circle has compactness = 1. Irregular borders give higher values.

    Args:
        mask (np.ndarray): Binary lesion mask.

    Returns:
        float: Border irregularity score. Higher = more irregular = more concerning.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    contour = max(contours, key=cv2.contourArea)
    area      = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if area < 1e-6:
        return 0.0

    # Compactness = perimeter² / (4π × area)
    # Circle = 1.0, irregular shapes > 1.0
    compactness = (perimeter ** 2) / (4 * np.pi * area)
    return round(compactness, 4)


# ─── C: Colour Variation ───────────────────────────────────────────

def compute_colour(image_bgr: np.ndarray, mask: np.ndarray) -> float:
    """
    Measure colour variation within the lesion region.
    Uses standard deviation of pixel colours inside the lesion mask.

    Args:
        image_bgr (np.ndarray): Original BGR image.
        mask      (np.ndarray): Binary lesion mask.

    Returns:
        float: Colour variation score. Higher = more colour variation = more concerning.
    """
    # Extract lesion pixels only
    lesion_pixels = image_bgr[mask > 0]
    if len(lesion_pixels) == 0:
        return 0.0

    # Standard deviation across all colour channels
    std_per_channel = np.std(lesion_pixels.astype(float), axis=0)
    colour_score    = float(np.mean(std_per_channel)) / 255.0

    return round(colour_score, 4)


# ─── D: Diameter ───────────────────────────────────────────────────

def compute_diameter(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    contour     = max(contours, key=cv2.contourArea)
    _, _, w, h  = cv2.boundingRect(contour)

    img_h, img_w = mask.shape
    diameter     = max(w, h) / max(img_w, img_h)

    # Sanity check: if lesion covers more than 90% of image
    # it is almost certainly a mask extraction failure
    # Cap at 0.9 to prevent misleading D scores
    diameter = min(diameter, 0.9)

    return round(diameter, 4)


# ─── Lesion Mask Extraction ────────────────────────────────────────

def extract_lesion_mask(image_bgr: np.ndarray) -> np.ndarray:
    """
    Extract a binary mask of the lesion region using colour thresholding.
    Works on both dermoscopic and clinical images.

    Args:
        image_bgr (np.ndarray): Input BGR image.

    Returns:
        np.ndarray: Binary mask (255 = lesion, 0 = background).
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Otsu thresholding — automatically finds optimal threshold
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological cleanup — remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    return mask


# ─── Main ABCD Function ────────────────────────────────────────────

def compute_abcd_scores(image_bgr: np.ndarray) -> np.ndarray:
    """
    Compute all 4 ABCD scores for a skin lesion image.

    Args:
        image_bgr (np.ndarray): Input BGR image from OpenCV.

    Returns:
        numpy.ndarray: Shape [4] — [asymmetry, border, colour, diameter]
    """
    mask = extract_lesion_mask(image_bgr)

    asymmetry = compute_asymmetry(mask)
    border    = compute_border(mask)
    colour    = compute_colour(image_bgr, mask)
    diameter  = compute_diameter(mask)

    scores = np.array([asymmetry, border, colour, diameter], dtype=np.float32)

    print(f"  [ABCD Scores]   A={asymmetry}  B={border}  C={colour}  D={diameter}")

    return scores