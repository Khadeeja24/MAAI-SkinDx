# MAAI-SkinDx

**Multi-Agent AI Framework for Automated Skin Disease Diagnosis**  
University of Galway | Supervisor: Dr. Saeed Alsamhi

---

## Overview

MAAI-SkinDx is a multi-agent AI system for clinical-grade skin disease diagnosis from dermoscopy images. The system uses nine specialist agents coordinated through an Adaptive Coordination Intelligence layer that manages agent trust, resolves conflicts, fuses evidence, and produces structured clinical reports with XAI explanations.

The system classifies 9 skin disease classes and 5 pox disease classes, handles out-of-distribution (OOD) detection, and supports both image-only and image+symptom workflows.

---

## Pipeline Architecture

```
Image
  │
  ▼
Agent 1 — Image Quality Gate
  │ (PASS / REJECT)
  ▼
Agent 2 — Visual Feature Extraction  (6916-dim vector)
  │
  ▼
Agent 3 — Disease Classifier  (9 classes)
  │
  ├──► Pox Classifier MLP  (runs on every case — 5 pox classes)
  │         │
  │    Four-way routing:
  │    A: Agent 3 confident, Pox not → normal pipeline
  │    B: Agent 3 confident, Pox also → Pox overrides (softmax fix)
  │    C: Agent 3 uncertain, Pox confident → Pox confirmed
  │    D: Agent 3 uncertain, Pox uncertain → TRUE OOD WARNING
  │
  ▼
Adaptive Coordination Intelligence (Phases 1–10)
  │
  ├── Phase 1  — Unified message format
  ├── Phase 2  — Agent assessment (base trust)
  ├── Phase 3  — Parallel: Agent 4 + Clinical Agent
  ├── Phase 4  — Adaptive trust per case
  ├── Phase 5  — Adaptive fusion (trust-weighted)
  ├── Phase 6a — Pre-fusion reasoning
  ├── Phase 7  — Conflict resolution (4 conflict types)
  ├── Phase 6b — Post-fusion reasoning
  ├── Phase 8  — Decision intelligence (9 decision types)
  ├── Phase 9  — Agent 7 clinical report generation
  └── Phase 10 — Agent 6 background learning (EMA trust update)
```

---

## Agents

| Agent | Role | Key Result |
|---|---|---|
| Agent 1 | Image Quality Gate | F1 97.89%, AUC-ROC 0.9984 |
| Agent 2 | Visual Feature Extraction | 6916-dim vector (DINOv2 + Derm Foundation + ABCD) |
| Agent 3 | Disease Classifier (9 classes) | Macro F1 74.18%, Acc 76.93% |
| Pox Classifier | Pox Detection MLP (5 classes) | Softmax overconfidence fix |
| Agent 4 | XAI — Grad-CAM + LLM Explanation | Clinician score 4.105/5 |
| Clinical Agent | Symptom Risk Scoring (RandomForest) | F1 92.68% (PAD-UFES-20) |
| Agent 5 | RAG Knowledge Retrieval | ChromaDB vector store |
| Agent 6 | Background Learning (EMA Trust) | TrustMAE 0.0739 |
| Agent 7 | Expert Clinical Report (LLM) | Clinician score 4.085/5 |
| **Orchestrator** | **Adaptive Coordination (10 Phases)** | **System Macro F1 75.11%** |

---

## Coordination Agent — Key Results

Evaluated against the 19-step Strong Coordination Agent Evaluation Guide.

| Metric | Value |
|---|---|
| System Macro F1 | 75.11% |
| AUC-ROC | 0.9893 |
| AUPRC | 0.9741 |
| Malignancy Coverage | 99.8% |
| ECE (Calibration) | 0.0474 |
| Conflict Resolution Rate (natural conflicts) | 99.82% |
| Malignancy Sensitivity in Conflicts | 100% |
| Agreement Gain (pre → post coordination) | +16.57pp (NAG = 0.44) |
| Reasoning Gain (Phase 6+7 contribution) | +2.70pp |
| Regret Reduction vs Solo Agent 3 | 94.9% |
| Recovery Time after agent degradation | 12 cases |
| Statistical Significance (McNemar's test) | p < 0.0001 |
| 95% CI vs best baseline | [4.35%, 9.31%] |

---

## Baseline Comparison

| Method | Macro F1 |
|---|---|
| Majority Voting | 70.43% |
| Static Weighted | 71.87% |
| Dempster-Shafer | 72.55% |
| Solo Agent 3 | 74.18% |
| **MAAI-SkinDx (Adaptive Coordination)** | **75.11%** |

---

## Disease Classes

**Skin (9 classes — HAM10000 / ISIC):**  
Melanoma (MEL) · Nevus (NV) · Basal Cell Carcinoma (BCC) · Actinic Keratosis/SCC (AK) · Benign Keratosis (BKL) · Vascular Lesion (VASC) · Dermatofibroma (DF) · Infection (INF) · Eczema (ECZ)

**Pox (5 classes — pox dataset):**  
Monkeypox (MPOX) · Varicella-Zoster (VZV) · Molluscum Contagiosum (MCV) · Cowpox (CPXV) · Hand-Foot-Mouth Disease (HFMD)

---

## Datasets

| Dataset | Images | Classes | Role |
|---|---|---|---|
| HAM10000 | 10,015 | 7 | Agent 3 train/test |
| Fitzpatrick-17k | 15,500 | 114 | Fairness evaluation |
| PAD-UFES-20 | 2,298 | 6 | Clinical Agent training |
| Massive Balanced | 262,874 | 34 | Agent 2 feature evaluation |
| DDI-31 | 647 | 31 | Agent 3 OOD test only |
| Pox Dataset | 6,356 | 5 | Pox Classifier training |

---

## Agent 1 — Image Quality Gate

Three sequential checks:
1. **Blur Detection** — Laplacian variance, threshold = 100
2. **Quality Assessment** — BRISQUE score, threshold = 40
3. **Skin Detection** — ResNet-50 fine-tuned binary classifier

Model comparison: MobileNetV3 vs EfficientNet-B0 vs ResNet-50  
Winner: ResNet-50 — lowest false positive rate on held-out test  
Results: F1 97.89% | AUC-ROC 0.9984 | Threshold 0.8783  
Fairness: Tested across Fitzpatrick skin tone groups — ACCEPTABLE

---

## Agent 2 — Visual Feature Extraction

Three frozen streams → 6916-dim combined feature vector:

| Stream | Model | Dimensions |
|---|---|---|
| Dermoscopic | DINOv2 ViT-B/14 (Meta AI) | 768 |
| Clinical | Derm Foundation BiT (Google) | 6144 |
| ABCD Scores | OpenCV (mathematical) | 4 |
| **Combined** | **Equal weighted** | **6916** |

---

## Agent 3 — Disease Classifier

Fine-tuned ResNet-50 on HAM10000 + multi-dataset extension.  
9-class skin disease classification with OOD threshold = 0.40.

Model comparison: EfficientNet-B3 vs ResNet-50 vs DenseNet-121  
Winner: ResNet-50 (CV F1 61.43% ± 1.76%)  
Final Results: Macro F1 74.18% | Accuracy 76.93% | DDI-31 OOD F1 52.29%

---

## Project Structure

```
MAAI-SkinDx/
├── agents/
│   ├── agent1_image_quality/
│   ├── agent2_feature_extraction/
│   ├── agent3_diagnosis/
│   ├── agent4_xai/
│   ├── agent5_rag/
│   ├── agent6_background/
│   ├── agent7_Expert/
│   ├── clinical_agent/
│   └── pox_classifier/
├── orchestrator/
│   ├── orchestrator.py
│   ├── evaluate_coord_agent.py
│   └── eval_results/
├── evaluation/
├── scripts/
│   ├── evaluate_ablation.py
│   ├── evaluate_baseline.py
│   ├── evaluate_calibration.py
│   ├── evaluate_conflict_resolution.py
│   ├── evaluate_missing_agent.py
│   ├── evaluate_significance.py
│   ├── train_pox_mlp.py
│   └── ...
├── data/
│   ├── processed/
│   └── agent6/
└── notebooks/
```

---

## Setup

```bash
conda activate maai_skin
```

**Run full pipeline:**
```bash
python notebooks/test_orchestrator.py
```

**Run coordination agent evaluation (19 steps):**
```bash
cd orchestrator
python evaluate_coord_agent.py
```

**Train Pox Classifier:**
```bash
python scripts/train_pox_mlp.py
```

---

## Tech Stack

Python · PyTorch · TensorFlow · timm · scikit-learn · OpenCV · HuggingFace · ChromaDB · Groq (LLaMA-3.3-70b) · python-docx · matplotlib

---

## Citation

If you use this work, please cite:

```
MAAI-SkinDx: Multi-Agent AI Framework for Automated Skin Disease Diagnosis
University of Galway, 2024
Supervisor: Dr. Saeed Alsamhi
```
