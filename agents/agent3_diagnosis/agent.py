# MAAI-SkinDx | Agent 3 — Disease Classification Agent
# Uses CombinedModel — same architecture as training.
# Receives Agent 2 features from Orchestrator.

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision import transforms, models
from PIL import Image as PILImage
from .config import (
    DEVICE, MODEL_PATH, NUM_CLASSES,
    CLASS_NAMES, CLASS_FULL_NAMES,
    MELANOMA_SAFETY_THRESHOLD, TOP_N, IMG_SIZE
)

BACKBONE_DIM = 2048
AGENT2_DIM   = 6916

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]),
])


class CombinedModel(nn.Module):
    """Identical to train_agent3_final.py — must match exactly."""
    def __init__(self, combined_dim, num_classes, use_agent2):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4
        self.avgpool = resnet.avgpool
        self.use_agent2 = use_agent2
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(combined_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, agent2_feats=None):
        x = self.conv1(images)
        x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        backbone_feats = torch.flatten(x, 1)
        if self.use_agent2 and agent2_feats is not None:
            combined = torch.cat([backbone_feats, agent2_feats], dim=1)
        else:
            combined = backbone_feats
        return self.classifier(combined), backbone_feats


class DiagnosisAgent:

    def __init__(self):
        self.name         = "Agent 3 — Disease Classification Agent"
        self.model        = None
        self.use_agent2   = False
        self.combined_dim = BACKBONE_DIM
        print(f"\n[{self.name}] Initialising ...")
        self._load_model()
        print(f"[{self.name}] Ready")

    def _load_model(self):
        script_dir   = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        model_path   = os.path.join(project_root, MODEL_PATH)

        if not os.path.exists(model_path):
            print(f"  [Agent 3] WARNING: {model_path} not found")
            print(f"  Run scripts/train_agent3_final.py first")
            return

        try:
            ckpt = torch.load(model_path, map_location=DEVICE,
                              weights_only=False)
            use_agent2   = ckpt.get("use_agent2", False)
            combined_dim = ckpt.get("combined_dim", BACKBONE_DIM)
            num_classes  = ckpt.get("num_classes", NUM_CLASSES)

            self.use_agent2   = use_agent2
            self.combined_dim = combined_dim

            self.model = CombinedModel(
                combined_dim=combined_dim,
                num_classes=num_classes,
                use_agent2=use_agent2)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.model = self.model.to(DEVICE)

            print(f"  [Agent 3] Loaded from {model_path}")
            print(f"  [Agent 3] Architecture : "
                  f"{'COMBINED' if use_agent2 else 'BACKBONE'} "
                  f"{combined_dim}-dim")
            print(f"  [Agent 3] Hold-out F1  : "
                  f"{ckpt.get('val_f1', 0)*100:.2f}%")
            print(f"  [Agent 3] Classes      : {num_classes}")

            # GPU warm-up — eliminates slow first inference
            # PyTorch compiles CUDA kernels on first forward pass.
            # Running a dummy image now means first real patient
            # image runs at full speed (~6s instead of ~17s).
            print(f"  [Agent 3] Warming up GPU...")
            with torch.no_grad():
                dummy_img = torch.zeros(
                    1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
                dummy_a2  = torch.zeros(
                    1, AGENT2_DIM).to(DEVICE) \
                    if use_agent2 else None
                self.model(dummy_img, dummy_a2)
            print(f"  [Agent 3] GPU ready")

        except Exception as e:
            print(f"  [Agent 3] Load failed: {e}")
            self.model = None

    def _predict(self, image_path, a2_tensor):
        if self.model is None:
            return np.ones(NUM_CLASSES) / NUM_CLASSES
        try:
            img    = PILImage.open(image_path).convert("RGB")
            tensor = VAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out, _ = self.model(tensor, a2_tensor)
                probs  = torch.softmax(out, dim=1)
            return probs.squeeze().cpu().numpy()
        except Exception as e:
            print(f"  [Agent 3] Prediction error: {e}")
            return np.ones(NUM_CLASSES) / NUM_CLASSES

    def _backbone_features(self, image_path, a2_tensor):
        if self.model is None:
            return np.zeros(BACKBONE_DIM, dtype=np.float32)
        try:
            img    = PILImage.open(image_path).convert("RGB")
            tensor = VAL_TRANSFORM(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, feats = self.model(tensor, a2_tensor)
            return feats.squeeze().cpu().numpy()
        except Exception as e:
            print(f"  [Agent 3] Backbone error: {e}")
            return np.zeros(BACKBONE_DIM, dtype=np.float32)

    def run(self, image_path: str,
            agent2_features=None) -> dict:

        print(f"\n{'='*60}")
        print(f"  {self.name}")
        print(f"  Image : {image_path}")
        print(f"{'='*60}")

        if self.model is None:
            return {
                "status"           : "FAIL",
                "reason"           : "Model not loaded.",
                "top_5_predictions": [],
                "backbone_features": None,
                "agent2_features"  : agent2_features,
            }

        # Prepare Agent 2 tensor
        a2_tensor = None
        if self.use_agent2 and agent2_features is not None:
            try:
                a2_np = np.asarray(agent2_features, dtype=np.float32)
                if a2_np.shape[0] == AGENT2_DIM:
                    a2_tensor = torch.from_numpy(a2_np)\
                        .unsqueeze(0).to(DEVICE)
            except Exception as e:
                print(f"  [Agent 3] Agent2 prep error: {e}")

        # Predict
        print(f"\n  -- Classification --")
        probs = self._predict(image_path, a2_tensor)

        # Melanoma safety threshold
        melanoma_prob    = float(probs[0])
        melanoma_flagged = melanoma_prob >= MELANOMA_SAFETY_THRESHOLD
        predicted_idx    = 0 if melanoma_flagged else int(np.argmax(probs))

        if melanoma_flagged:
            print(f"  WARNING: Melanoma safety threshold triggered "
                  f"(prob={melanoma_prob:.3f} >= {MELANOMA_SAFETY_THRESHOLD})")

        # Top 5
        top_indices = np.argsort(probs)[::-1][:TOP_N]
        top_5 = []
        for rank, idx in enumerate(top_indices):
            idx  = int(idx)
            name = CLASS_NAMES.get(idx, f"Class {idx}")
            top_5.append({
                "rank"          : rank + 1,
                "class_index"   : idx,
                "class_name"    : name,
                "class_full"    : CLASS_FULL_NAMES.get(idx, name),
                "confidence"    : float(probs[idx]),
                "confidence_pct": f"{probs[idx]*100:.1f}%",
            })

        print(f"\n  Top {TOP_N} predictions:")
        for p in top_5:
            bar = "█" * int(p["confidence"] * 20)
            print(f"    {p['rank']}. {p['class_name']:<45} "
                  f"{p['confidence_pct']:>6}  {bar}")

        print(f"\n  -- Backbone features for Agent 4 --")
        backbone = self._backbone_features(image_path, a2_tensor)
        print(f"  Shape: {backbone.shape}")
        print(f"{'='*60}\n")

        predicted_name = CLASS_NAMES.get(predicted_idx,
                                          f"Class {predicted_idx}")
        return {
            "status"           : "PASS",
            "reason"           : "Classification complete.",
            "predicted_class"  : predicted_name,
            "predicted_name"   : CLASS_FULL_NAMES.get(
                predicted_idx, predicted_name),
            "predicted_index"  : predicted_idx,
            "confidence"       : float(probs[predicted_idx]),
            "confidence_pct"   : f"{probs[predicted_idx]*100:.1f}%",
            "melanoma_flagged" : melanoma_flagged,
            "melanoma_prob"    : melanoma_prob,
            "top_5_predictions": top_5,
            "all_probabilities": {
                CLASS_NAMES.get(i, f"Class {i}"): float(probs[i])
                for i in range(NUM_CLASSES)
            },
            "backbone_features": backbone,
            "agent2_features"  : agent2_features,
        }