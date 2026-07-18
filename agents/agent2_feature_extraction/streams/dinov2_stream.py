# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 2 — Stream A: Dermoscopic Features
# Feature Extractor: DINOv2 ViT-B/14
# ══════════════════════════════════════════════════════════════════
# Selected by k-NN probe comparison on HAM10000.
# k-NN F1: 45.46% — best among all four models tested.
#
# DINOv2 (Meta AI) pretrained on 142M curated images using
# self-supervised DINO objective. Features are specifically
# optimized for k-NN retrieval — a stated design goal of DINOv2.
#
# Why DINOv2 over PanDerm (dermoscopic specialist):
#   PanDerm uses contrastive pretraining optimized for visual
#   similarity, not disease boundary classification.
#   DINOv2 DINO objective produces better k-NN separable
#   features regardless of domain.
#
# Output: 768-dimensional frozen feature vector.
# ══════════════════════════════════════════════════════════════════

import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import timm
import os
import sys
import cv2

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..config import DEVICE, DINOV2_MODEL_NAME, DINOV2_EMBED_DIM, \
    NORMALIZE_MEAN, NORMALIZE_STD, IMAGE_SIZE

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])


class DINOv2Stream:

    def __init__(self):
        self.model     = None
        self.embed_dim = DINOV2_EMBED_DIM
        self._load_model()

    def _load_model(self):
        try:
            self.model = timm.create_model(
                DINOV2_MODEL_NAME,
                pretrained=True,
                num_classes=0,
                img_size=IMAGE_SIZE
            )
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
            self.model = self.model.to(DEVICE)
            print(f"  [DINOv2] Loaded successfully on {DEVICE}")
        except Exception as e:
            print(f"  [DINOv2] Load failed: {e}")
            print(f"  [DINOv2] Returning zero embedding as fallback")
            self.model = None

    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract 768-dim DINOv2 embedding from a BGR OpenCV image.

        Args:
            image (np.ndarray): BGR image from OpenCV.

        Returns:
            np.ndarray: 768-dim feature vector.
        """
        if self.model is None:
            return np.zeros(self.embed_dim, dtype=np.float32)

        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            tensor    = TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                embedding = self.model(tensor)

            embedding = embedding.squeeze().cpu().numpy()
            print(f"  [DINOv2] Embedding shape: {embedding.shape}")
            return embedding

        except Exception as e:
            print(f"  [DINOv2] Extraction error: {e}")
            return np.zeros(self.embed_dim, dtype=np.float32)