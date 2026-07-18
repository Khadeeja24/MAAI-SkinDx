# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 2 — Stream B: Derm Foundation
# ══════════════════════════════════════════════════════════════════
# Google's Derm Foundation pretrained on 4.5M clinical skin images.
# BiT-101x3 CNN architecture. TensorFlow SavedModel.
# Used completely FROZEN — no training, no fine-tuning.
# Extracts 6144-dimensional embedding from the input image.
# ══════════════════════════════════════════════════════════════════

import numpy as np
import cv2
from PIL import Image
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..config import DERM_FOUNDATION_MODEL_NAME, DERM_FOUNDATION_EMBED_DIM


class DermFoundationStream:
    """
    Stream B — Derm Foundation Feature Extractor.
    Loads Google Derm Foundation TensorFlow SavedModel.
    Freezes all weights. Extracts 6144-dim embeddings.
    """

    def __init__(self):
        print(f"  [DermFoundation] Loading {DERM_FOUNDATION_MODEL_NAME} ...")

        try:
            import tensorflow as tf
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(DERM_FOUNDATION_MODEL_NAME)
            self.model = tf.saved_model.load(model_path)
            self.infer = self.model.signatures["serving_default"]
            self.tf    = tf

            print(f"  [DermFoundation] Loaded successfully")

        except Exception as e:
            print(f"  [DermFoundation] WARNING: Could not load — {e}")
            print(f"  [DermFoundation] Falling back to zero embedding ({DERM_FOUNDATION_EMBED_DIM}D)")
            self.model = None
            self.infer = None
            self.tf    = None

    def extract(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Extract Derm Foundation embedding from a BGR image.

        Returns:
            numpy.ndarray: Shape [6144]
        """
        if self.model is None:
            print(f"  [DermFoundation] Using zero fallback embedding")
            return np.zeros(DERM_FOUNDATION_EMBED_DIM, dtype=np.float32)

        try:
            import io

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)

            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()

            example = self.tf.train.Example(
                features=self.tf.train.Features(
                    feature={
                        'image/encoded': self.tf.train.Feature(
                            bytes_list=self.tf.train.BytesList(value=[image_bytes])
                        )
                    }
                )
            ).SerializeToString()

            input_tensor = self.tf.constant([example], dtype=self.tf.string)
            output       = self.infer(inputs=input_tensor)
            embedding    = output['embedding'].numpy().squeeze()

            print(f"  [DermFoundation] Embedding shape: {embedding.shape}")
            return embedding.astype(np.float32)

        except Exception as e:
            print(f"  [DermFoundation] Extraction error: {e} — using zero fallback")
            return np.zeros(DERM_FOUNDATION_EMBED_DIM, dtype=np.float32)