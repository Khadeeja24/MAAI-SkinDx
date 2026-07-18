# MAAI-SkinDx
### Multi-Agent AI Framework for Automated Skin Disease Detection
**University of Galway | Supervisor: Dr. Saeed Alsamhi**

---

## Project Overview
A multi-agent AI pipeline that takes a patient skin image and produces
a clinical diagnosis report with XAI explanations.

## Pipeline Architecture

| Agent | Role | Status |
|-------|------|--------|
| Agent 1 | Image Quality Gate | ✅ Complete |
| Agent 2 | Visual Feature Extraction | ✅ Complete |
| Agent 3 | Disease Classification | 🔄 In Progress |
| Agent 4 | XAI — Grad-CAM + SHAP | 📋 Planned |
| Clinical Agent | Symptom Risk Scoring | 📋 Planned |
| Agent 5 | RAG Knowledge Retrieval | 📋 Planned |
| Agent 7 | Clinical Report Generation | 📋 Planned |

---

## Agent 1 — Image Quality Agent ✅
Three sequential checks:
- Blur Detection (Laplacian variance, threshold=100)
- Quality Assessment (BRISQUE score, threshold=40)
- Skin Detection (ResNet-50 fine-tuned binary classifier)

**Model comparison:** MobileNetV3 vs EfficientNet-B0 vs ResNet-50  
**Winner:** ResNet-50 — lowest false positive count on held-out test  
**Results:** Test F1 97.89% | AUC-ROC 0.9984 | Threshold 0.8783  
**Fairness:** Tested across Fitzpatrick skin tone groups — ACCEPTABLE

---

## Agent 2 — Visual Feature Extraction Agent ✅
Three frozen streams → 6916-dim combined feature vector:

| Stream | Model | Dimensions | Selection F1 |
|--------|-------|-----------|--------------|
| A — Dermoscopic | DINOv2 ViT-B/14 (Meta AI) | 768 | k-NN F1 45.46% |
| B — Clinical | Derm Foundation BiT (Google) | 6144 | F1 76.83% |
| C — ABCD Scores | OpenCV (mathematical) | 4 | — |
| **Combined** | Equal weighted (×1/3 each) | **6916** | — |

**Model comparison — Dermoscopic:** PanDerm vs ViT-B/16 vs DINOv2 vs ResNet-50  
**Model comparison — Clinical:** ResNet-50 vs EfficientNet-B4 vs Derm Foundation

---

## Agent 3 — Disease Classification 🔄
Fine-tuned CNN on HAM10000 (7 disease classes)

**Model comparison:** EfficientNet-B3 vs ResNet-50 vs DenseNet-121  
**Dataset:** HAM10000 (10,015 images, split by lesion_id)  
**Winner:** ResNet-50 (CV F1 61.43% ± 1.76%)

---

## Datasets

| Dataset | Images | Classes | Role |
|---------|--------|---------|------|
| HAM10000 | 10,015 | 7 | Train + Test |
| Fitzpatrick-17k | 15,500 | 114 | Train (fairness) |
| PAD-UFES-20 | 2,298 | 6 | Train (symptoms) |
| Massive Balanced | 262,874 | 34 | Train (Phase 2) |
| DDI-31 | 647 | 31 | TEST ONLY |

---

## Tech Stack
Python · PyTorch · TensorFlow · timm · scikit-learn · OpenCV · HuggingFace

## Run
```bash
conda activate maai_skin
python notebooks/test_orchestrator.py
```