# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 2 — Configuration
# ══════════════════════════════════════════════════════════════════
# Model selection based on linear probe + k-NN probe evaluation:
#
#   Dermoscopic stream: DINOv2 ViT-B/14
#     Evaluated on HAM10000 (10,015 images, 7 disease classes)
#     Split by lesion_id (StratifiedGroupKFold — no data leakage)
#     k-NN F1: DINOv2 45.46% | ResNet-50 38.68% | ViT-B/16 37.47%
#              PanDerm 26.29% (worst — contrastive objective mismatch)
#
#   Clinical stream: Derm Foundation BiT
#     Evaluated on Fitzpatrick-17k (15,504 images, 3 classes)
#     F1: Derm Foundation 76.83% | EfficientNet-B4 57.62% | ResNet-50 57.41%
#     +19% margin — domain-specific pretraining clearly superior
#
# ══════════════════════════════════════════════════════════════════

import torch

# ─── Device ───────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Image settings ───────────────────────────────────────────────
IMAGE_SIZE     = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ─── Stream A: Dermoscopic — DINOv2 ViT-B/14 ──────────────────────
DERM_STREAM_MODEL  = "DINOv2"
DINOV2_MODEL_NAME  = "vit_base_patch14_dinov2.lvd142m"
DINOV2_EMBED_DIM   = 768

# ─── Stream B: Clinical — Derm Foundation BiT ─────────────────────
CLIN_STREAM_MODEL          = "DermFoundation"
DERM_FOUNDATION_MODEL_NAME = "google/derm-foundation"
DERM_FOUNDATION_EMBED_DIM  = 6144

# ─── Stream C: ABCD Scores ────────────────────────────────────────
ABCD_DIM = 4

# ─── Combined output ──────────────────────────────────────────────
# 768 (DINOv2) + 6144 (Derm Foundation) + 4 (ABCD) = 6916
COMBINED_FEATURE_DIM = DINOV2_EMBED_DIM + DERM_FOUNDATION_EMBED_DIM + ABCD_DIM