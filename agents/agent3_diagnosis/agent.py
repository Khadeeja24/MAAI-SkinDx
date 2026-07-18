# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 3 — Disease Classification Agent
# ══════════════════════════════════════════════════════════════════
# Receives from Orchestrator:
#   image_path      : path to approved image (from Agent 1)
#   agent2_features : 6916-dim feature vector (from Agent 2)
#
# Architecture:
#   ResNet-50 backbone fine-tuned on 4 datasets:
#     HAM10000 + Fitzpatrick-17k + PAD-UFES-20 + MassiveBalanced
#   Output: 9 unified disease classes
#   Backbone features (2048-dim) passed to Agent 4 for Grad-CAM
#
# Model performance:
#   Hold-out F1  : 74.57% (macro, 8,932 images)
#   Hold-out Acc : 76.94%
#   Best epoch   : 21
#
# Unified 9-class taxonomy:
#   0: Melanoma
#   1: Melanocytic Nevus
#   2: Basal Cell Carcinoma
#   3: Actinic Keratosis / Squamous Cell Carcinoma
#   4: Benign Keratosis
#   5: Vascular Lesion
#   6: Dermatofibroma
#   7: Inflammatory
#   8: Eczema / Dermatitis
#
# Clinical safety:
#   Melanoma safety threshold applied — if melanoma probability
#   exceeds 0.35, flagged as melanoma even if not top prediction.
#   Improves melanoma recall toward the 85% clinical target.
# ══════════════════════════════════════════════════════════════════

import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision import transforms, models
from PIL import Image as PILImage
from .config import (
    DEVICE, MODEL_PATH, IN_FEATURES, NUM_CLASSES,
    CLASS_NAMES, CLASS_FULL_NAMES,
    MELANOMA_SAFETY_THRESHOLD, TOP_N, IMG_SIZE
)

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    ),
])


class DiagnosisAgent:

    def __init__(self):
        self.name     = "Agent 3 — Disease Classification Agent"
        self.model    = None
        self.backbone = None

        print(f"\n[{self.name}] Initialising ...")
        self._load_model()
        print(f"[{self.name}] Ready")

    # ─── Private: build ResNet-50 architecture ─────────────────────

    def _build_resnet50(self):
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(IN_FEATURES, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )
        return model

    # ─── Private: load saved checkpoint ───────────────────────────

    def _load_model(self):
        script_dir   = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        model_path   = os.path.join(project_root, MODEL_PATH)

        if not os.path.exists(model_path):
            print(f"  [Agent 3] WARNING: Model not found at {model_path}")
            print(f"  [Agent 3] Run scripts/train_agent3_final.py first")
            return

        try:
            ckpt = torch.load(
                model_path, map_location=DEVICE,
                weights_only=False)

            self.model = self._build_resnet50()
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.model = self.model.to(DEVICE)

            # Backbone — ResNet-50 without the fc head
            # Used for feature extraction for Agent 4 Grad-CAM
            self.backbone = nn.Sequential(
                *list(self.model.children())[:-1],
                nn.Flatten()
            ).to(DEVICE)
            self.backbone.eval()

            saved_f1 = ckpt.get("val_f1", 0)
            saved_cls = ckpt.get("class_names", CLASS_NAMES)
            saved_datasets = ckpt.get("datasets_used", [])

            print(f"  [Agent 3] Model loaded from {model_path}")
            print(f"  [Agent 3] Hold-out F1  : {saved_f1*100:.2f}%")
            print(f"  [Agent 3] Classes      : {NUM_CLASSES}")
            print(f"  [Agent 3] Trained on   : "
                  f"{', '.join(saved_datasets)}")

        except Exception as e:
            print(f"  [Agent 3] Load failed: {e}")
            self.model    = None
            self.backbone = None

    # ─── Private: extract backbone features ───────────────────────

    def _extract_backbone_features(
        self, image_path: str
    ) -> np.ndarray:
        """
        Extract 2048-dim ResNet-50 backbone features.
        These penultimate-layer features are passed to
        Agent 4 for Grad-CAM explainability.
        """
        if self.backbone is None:
            return np.zeros(IN_FEATURES, dtype=np.float32)
        try:
            img    = PILImage.open(image_path).convert("RGB")
            tensor = VAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                feats = self.backbone(tensor)
            return feats.squeeze().cpu().numpy()
        except Exception as e:
            print(f"  [Agent 3] Backbone extraction error: {e}")
            return np.zeros(IN_FEATURES, dtype=np.float32)

    # ─── Private: get softmax probabilities ───────────────────────

    def _predict_probs(self, image_path: str) -> np.ndarray:
        """Get softmax probability for all 9 classes."""
        if self.model is None:
            return np.ones(NUM_CLASSES) / NUM_CLASSES
        try:
            img    = PILImage.open(image_path).convert("RGB")
            tensor = VAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = self.model(tensor)
                probs  = torch.softmax(logits, dim=1)
            return probs.squeeze().cpu().numpy()
        except Exception as e:
            print(f"  [Agent 3] Prediction error: {e}")
            return np.ones(NUM_CLASSES) / NUM_CLASSES

    # ─── Public: main run method ───────────────────────────────────

    def run(self, image_path: str,
            agent2_features: np.ndarray = None) -> dict:
        """
        Classify skin disease from approved image.
        Called by Orchestrator after Agent 2.

        Args:
            image_path      : path to image approved by Agent 1
            agent2_features : 6916-dim vector from Agent 2
                              (stored, passed to Agent 4 later)

        Returns:
            dict with:
                status              : "PASS" or "FAIL"
                predicted_class     : top predicted class name
                predicted_name      : full class name with description
                confidence          : confidence score (0-1)
                top_5_predictions   : list of top 5 with confidence
                all_probabilities   : all 9 class probabilities
                backbone_features   : 2048-dim for Agent 4 Grad-CAM
                agent2_features     : passed through from Agent 2
                melanoma_flagged    : True if safety threshold triggered
        """
        print(f"\n{'═'*60}")
        print(f"  {self.name}")
        print(f"  Image : {image_path}")
        print(f"{'═'*60}")

        if self.model is None:
            return {
                "status"          : "FAIL",
                "reason"          : (
                    "Agent 3 model not loaded. "
                    "Run scripts/train_agent3_final.py first."),
                "top_5_predictions" : [],
                "backbone_features" : None,
                "agent2_features"   : agent2_features,
            }

        # Step A — Get class probabilities
        print(f"\n  ── Classification ──")
        probs = self._predict_probs(image_path)

        # Step B — Melanoma safety threshold
        # Clinical safety: if melanoma probability exceeds threshold,
        # flag as melanoma even if another class scored higher.
        # Targets recall ≥ 85% for the most dangerous diagnosis.
        melanoma_prob    = float(probs[0])
        melanoma_flagged = False

        if melanoma_prob >= MELANOMA_SAFETY_THRESHOLD:
            predicted_idx    = 0
            melanoma_flagged = True
            print(f"  ⚠  Melanoma safety threshold triggered")
            print(f"     Melanoma prob={melanoma_prob:.3f} >= "
                  f"threshold={MELANOMA_SAFETY_THRESHOLD}")
        else:
            predicted_idx = int(np.argmax(probs))

        # Step C — Build top 5 predictions
        top_indices = np.argsort(probs)[::-1][:TOP_N]
        top_5       = []
        for rank, idx in enumerate(top_indices):
            idx      = int(idx)
            cls_name = CLASS_NAMES.get(idx, f"Class {idx}")
            top_5.append({
                "rank"          : rank + 1,
                "class_index"   : idx,
                "class_name"    : cls_name,
                "class_full"    : CLASS_FULL_NAMES.get(
                    idx, cls_name),
                "confidence"    : float(probs[idx]),
                "confidence_pct": f"{probs[idx]*100:.1f}%",
            })

        # Step D — Print top 5
        print(f"\n  Top {TOP_N} predictions:")
        for pred in top_5:
            flag = " ← FLAGGED" \
                if pred["class_index"] == 0 and melanoma_flagged \
                else ""
            bar = "█" * int(pred["confidence"] * 20)
            print(f"    {pred['rank']}. "
                  f"{pred['class_name']:<45} "
                  f"{pred['confidence_pct']:>6}  {bar}{flag}")

        # Step E — Extract backbone features for Agent 4
        print(f"\n  ── Backbone features for Agent 4 ──")
        backbone_feats = self._extract_backbone_features(image_path)
        print(f"  Backbone features : {backbone_feats.shape} "
              f"(for Grad-CAM in Agent 4)")

        print(f"{'═'*60}\n")

        predicted_name = CLASS_NAMES.get(predicted_idx,
                                          f"Class {predicted_idx}")

        return {
            "status"            : "PASS",
            "reason"            : "Classification complete.",
            "predicted_class"   : predicted_name,
            "predicted_name"    : CLASS_FULL_NAMES.get(
                predicted_idx, predicted_name),
            "predicted_index"   : predicted_idx,
            "confidence"        : float(probs[predicted_idx]),
            "confidence_pct"    : f"{probs[predicted_idx]*100:.1f}%",
            "melanoma_flagged"  : melanoma_flagged,
            "melanoma_prob"     : melanoma_prob,
            "top_5_predictions" : top_5,
            "all_probabilities" : {
                CLASS_NAMES.get(i, f"Class {i}"): float(probs[i])
                for i in range(NUM_CLASSES)
            },
            "backbone_features" : backbone_feats,
            "agent2_features"   : agent2_features,
        }