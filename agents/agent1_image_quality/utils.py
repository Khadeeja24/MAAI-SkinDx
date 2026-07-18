# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 1 — Image Quality Agent
# Utilities
# ══════════════════════════════════════════════════════════════════
# Shared helper functions used across all Agent 1 checks.
# ══════════════════════════════════════════════════════════════════

import cv2
import numpy as np


# ─── Image Loading ─────────────────────────────────────────────────

def load_image(image_path: str):
    """
    Load an image from disk using OpenCV.

    Args:
        image_path (str): Absolute or relative path to the image file.

    Returns:
        numpy.ndarray: BGR image array, or None if loading fails.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Utils] Failed to load image: {image_path}")
    return image


# ─── Image Resizing ────────────────────────────────────────────────

def resize_image(image: np.ndarray, max_size: int = 512) -> np.ndarray:
    """
    Resize image so its longest side equals max_size.
    Preserves aspect ratio. Reduces processing time for large images.

    Args:
        image    (np.ndarray): Input BGR image.
        max_size (int)       : Maximum allowed dimension. Default 512.

    Returns:
        numpy.ndarray: Resized image.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image

    scale = max_size / max(h, w)
    new_dimensions = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)


# ─── Colour Conversion ─────────────────────────────────────────────

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale.

    Args:
        image (np.ndarray): Input BGR image.

    Returns:
        numpy.ndarray: Single-channel grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)