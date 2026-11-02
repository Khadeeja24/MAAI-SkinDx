# Runs on EVERY case — not only OOD cases.
# This catches overconfident softmax cases where Agent 3 is
# confident but wrong (e.g. a pox image classified as melanoma
# at high confidence). Four outcomes are possible:
#   1. Agent 3 confident + pox below threshold → normal pipeline
#   2. Agent 3 confident + pox above threshold → pox overrides
#   3. Agent 3 uncertain + pox above threshold → pox confirmed
#   4. Agent 3 uncertain + pox below threshold → true OOD
#
# Input  : 772-dim features from Agent 2 vector
#           (DINOv2 768-dim + ABCD 4-dim, recovered from 6916-dim)
# Output : dict with pox_detected flag and confidence score.
#           pox_detected=False when below threshold.
#           Never returns None.

import os
import sys
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

MODEL_PATH = os.path.join(
    PROJECT_ROOT, "models", "pox_mlp", "pox_mlp.pth")

POX_CLASSES = {
    0: "Monkeypox",
    1: "Chickenpox",
    2: "Measles",
    3: "Cowpox",
    4: "HFMD",
}

# Confidence threshold — only fire if above this
POX_THRESHOLD = 0.65


class PoxMLP(nn.Module):
    def __init__(self, input_dim=772, num_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class PoxClassifierAgent:
    """
    Pox family disease classifier.
    Called by Orchestrator on EVERY case — not only OOD cases.
    Never called directly by any other agent.
    """

    def __init__(self):
        self.name      = "Pox Classifier Agent"
        self.model     = None
        self.device    = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_PATH):
            print(f"  [Pox] WARNING: model not found at {MODEL_PATH}")
            return
        try:
            ckpt       = torch.load(
                MODEL_PATH, map_location=self.device)
            self.model = PoxMLP(
                input_dim=772, num_classes=5).to(self.device)
            self.model.load_state_dict(ckpt["model_state"])
            self.model.eval()
            print(f"  [Pox] Loaded — val F1 "
                  f"{ckpt.get('val_f1', 0):.4f}")
        except Exception as e:
            print(f"  [Pox] Load error: {e}")
            self.model = None

    def run(self, agent2_feature_vector: np.ndarray) -> dict:
        """
        Classify pox disease from Agent 2 feature vector.

        The 6916-dim Agent 2 vector is structured as:
            [dinov2_norm * (1/3)] [derm_norm * (1/3)] [abcd_norm * (1/3)]
             768-dim               4096-dim             4-dim

        We recover DINOv2 and ABCD by multiplying back by 3,
        giving the same 772-dim input the MLP was trained on.

        Returns:
            dict with keys:
                status          : PASS or FAIL
                pox_detected    : True if confidence > threshold
                predicted_name  : pox disease name
                predicted_index : class index (0-4)
                confidence      : top-1 softmax probability
                all_probs       : dict of all class probabilities
        """
        if self.model is None:
            return {
                "status"        : "FAIL",
                "pox_detected"  : False,
                "predicted_name": None,
                "confidence"    : 0.0,
            }

        try:
            # Recover DINOv2 (first 768) and ABCD (last 4)
            # from the weighted 6916-dim Agent 2 vector
            dino_norm = agent2_feature_vector[:768] * 3.0
            abcd_norm = agent2_feature_vector[-4:]  * 3.0

            pox_feat  = np.concatenate(
                [dino_norm, abcd_norm]).astype(np.float32)

            tensor = torch.from_numpy(
                pox_feat).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(tensor)
                probs  = torch.softmax(logits, dim=1)
                probs  = probs.squeeze().cpu().numpy()

            pred_idx  = int(probs.argmax())
            pred_conf = float(probs[pred_idx])
            pred_name = POX_CLASSES[pred_idx]

            detected  = pred_conf >= POX_THRESHOLD

            print(f"  [Pox] {pred_name} "
                  f"({pred_conf*100:.1f}%) — "
                  f"{'DETECTED' if detected else 'below threshold'}")

            return {
                "status"         : "PASS",
                "pox_detected"   : detected,
                "predicted_name" : pred_name,
                "predicted_index": pred_idx,
                "confidence"     : pred_conf,
                "all_probs"      : {
                    POX_CLASSES[i]: float(probs[i])
                    for i in range(5)
                },
            }

        except Exception as e:
            print(f"  [Pox] Inference error: {e}")
            return {
                "status"       : "FAIL",
                "pox_detected" : False,
                "predicted_name": None,
                "confidence"   : 0.0,
            }