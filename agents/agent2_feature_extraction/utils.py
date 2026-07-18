# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 2 — Visual Feature Extraction Agent
# Utilities
# ══════════════════════════════════════════════════════════════════
# Shared helper functions used across all Agent 2 streams.
# ══════════════════════════════════════════════════════════════════

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from .config import IMAGE_SIZE, NORMALIZE_MEAN, NORMALIZE_STD

# ─── Image Loading ─────────────────────────────────────────────────

def load_image(image_path: str):
    """
    Load image from disk.

    Args:
        image_path (str): Path to image file.

    Returns:
        numpy.ndarray: BGR image, or None if loading fails.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Utils] Failed to load image: {image_path}")
    return image


# ─── Preprocessing for Pretrained Models ───────────────────────────

def preprocess_for_model(image_bgr: np.ndarray) -> torch.Tensor:
    """
    Convert BGR image to normalised tensor for pretrained models.
    Resizes to IMAGE_SIZE x IMAGE_SIZE, normalises with ImageNet stats.

    Args:
        image_bgr (np.ndarray): Input BGR image from OpenCV.

    Returns:
        torch.Tensor: Shape [1, 3, 224, 224], ready for model input.
    """
    # BGR → RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Convert to PIL
    pil_image = Image.fromarray(image_rgb)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])

    # Apply and add batch dimension
    tensor = transform(pil_image).unsqueeze(0)
    return tensor


# ─── Colour Conversion ─────────────────────────────────────────────

def to_grayscale(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)