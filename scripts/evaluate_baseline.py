# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Evaluation Step 1 — Baseline Comparison
# Document 1, Step 1 — Strong Coordination Agent Evaluation Guide
# ══════════════════════════════════════════════════════════════════
# Compares three systems on the held-out test set (7,734 images):
#
#   System 1 : Agent 3 alone (raw softmax argmax)
#   System 2 : Agent 3 + OOD detection
#   System 3 : Agent 3 + OOD + Coordination decision policy
#
# Uses SEED=42 and exact same split logic as train_agent3_final.py.
# Reconstructs the 7,734 test split without any retraining.
#
# Metrics per system:
#   Accuracy, Macro F1, Weighted F1, Precision, Recall
#   Per-class F1
#   OOD cases caught
#   Decision distribution
#   Ablation: OOD threshold sensitivity
# ══════════════════════════════════════════════════════════════════

import os
import sys
import json
import time
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime
from pathlib import Path
from PIL import Image as PILImage
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score,
)

# ── Paths ────────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

UNIFIED_CSV    = os.path.join(
    PROJECT_ROOT, "data", "processed",
    "agent3_unified_dataset.csv")
LABEL_MAP_CSV  = os.path.join(
    PROJECT_ROOT, "data", "processed",
    "agent3_label_mapping.csv")
CACHE_FEATURES = os.path.join(
    PROJECT_ROOT, "data", "processed",
    "agent2_cache_features.npy")
CACHE_INDEX    = os.path.join(
    PROJECT_ROOT, "data", "processed",
    "agent2_cache_index.csv")
MODEL_PATH     = os.path.join(
    PROJECT_ROOT, "models", "agent3",
    "agent3_resnet50_final.pth")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "evaluation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Config — must match training exactly ─────────────────────────
SEED              = 42
DEVICE            = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE          = 224
BACKBONE_DIM      = 2048
AGENT2_DIM        = 6916
MAX_PER_CLASS     = 6000
MAX_HAM_PER_CLASS = 2000
MALIGNANT_CLASSES = {0, 2, 3}
OOD_THRESHOLD     = 0.40
BATCH_SIZE        = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SEP  = "=" * 65
SEP2 = "-" * 65


# ══════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURE
# Verified against checkpoint keys:
#   classifier.1 : Linear(combined_dim, 512)
#   classifier.3 : BatchNorm1d(512)
#   classifier.5 : Linear(512, num_classes)
# ══════════════════════════════════════════════════════════════════

class CombinedModel(nn.Module):
    """
    Agent 3 model architecture.
    Backbone layers exposed with original ResNet50 names
    to match checkpoint keys exactly:
      conv1, bn1, layer1, layer2, layer3, layer4, avgpool
    Classifier verified against checkpoint:
      classifier.1 : Linear(combined_dim, 512)
      classifier.3 : BatchNorm1d(512)
      classifier.5 : Linear(512, num_classes)
    """

    def __init__(self, num_classes: int,
                 use_agent2: bool,
                 combined_dim: int):
        super().__init__()
        base = models.resnet50(weights=None)

        # Expose with original ResNet50 names
        # so checkpoint keys load without mismatch
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool

        self.use_agent2  = use_agent2

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),               # 0
            nn.Linear(combined_dim, 512),  # 1
            nn.ReLU(),                     # 2
            nn.BatchNorm1d(512),           # 3
            nn.ReLU(),                     # 4
            nn.Linear(512, num_classes),   # 5
        )

    def forward(self, img, agent2_feat=None):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)

        if self.use_agent2 and agent2_feat is not None:
            feat = torch.cat([feat, agent2_feat], dim=1)

        return self.classifier(feat)


# ══════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════

class EvalDataset(Dataset):
    """
    Evaluation dataset — no augmentation, VAL_TRANSFORM only.
    Returns image tensor, agent2 features, label, image path.
    """

    def __init__(self, df, use_agent2,
                 features_arr, path_to_idx):
        self.paths      = [
            os.path.normpath(p)
            for p in df["image_path"]]
        self.labels     = [
            int(l) for l in df["unified_label"]]
        self.use_agent2 = use_agent2
        self.features   = features_arr
        self.p2i        = path_to_idx
        self.transform  = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path  = self.paths[idx]
        label = self.labels[idx]

        try:
            img = PILImage.open(path).convert("RGB")
            img = self.transform(img)
        except Exception:
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)

        if self.use_agent2 and self.features is not None:
            cidx = self.p2i.get(path, -1)
            a2   = (
                torch.from_numpy(
                    self.features[cidx].copy()).float()
                if cidx >= 0
                else torch.zeros(AGENT2_DIM)
            )
        else:
            a2 = torch.zeros(AGENT2_DIM)

        return img, a2, label, path


# ══════════════════════════════════════════════════════════════════
# STEP 1 — Reproduce exact test split
# Identical capping + split logic as train_agent3_final.py
# ══════════════════════════════════════════════════════════════════

def reproduce_test_split():
    print(f"\n{SEP}")
    print(f"  Step 1 — Reproducing Exact Test Split "
          f"(Seed={SEED})")
    print(SEP)

    df = pd.read_csv(UNIFIED_CSV)
    df["exists"] = df["image_path"].apply(os.path.exists)
    df = df[df["exists"]].copy().reset_index(drop=True)
    df["image_path"] = df["image_path"].apply(
        os.path.normpath)

    label_map  = pd.read_csv(LABEL_MAP_CSV)
    class_names = {
        int(r["class_index"]): r["class_name"]
        for _, r in label_map.iterrows()
        if int(r["class_index"]) in df["unified_label"].unique()
    }
    num_classes = len(class_names)

    print(f"  Valid images : {len(df):,}")
    print(f"  Classes      : {num_classes}")

    # Class capping — identical to training
    capped_rows = []
    for cls_idx in sorted(df["unified_label"].unique()):
        cls_df = df[df["unified_label"] == cls_idx].copy()

        if cls_idx in MALIGNANT_CLASSES:
            capped_rows.append(cls_df)
            continue

        ham_rows  = cls_df[cls_df["source"] == "HAM10000"]
        rest_rows = cls_df[cls_df["source"] != "HAM10000"]

        if len(ham_rows) > MAX_HAM_PER_CLASS:
            ham_rows = ham_rows.sample(
                n=MAX_HAM_PER_CLASS, random_state=SEED)

        remaining = MAX_PER_CLASS - len(ham_rows)
        if len(rest_rows) > remaining:
            rest_rows = rest_rows.sample(
                n=remaining, random_state=SEED)

        capped_rows.append(
            pd.concat([ham_rows, rest_rows]))

    df_capped = pd.concat(
        capped_rows).reset_index(drop=True)
    print(f"  After capping: {len(df_capped):,}")

    # Train/test split — identical to training
    ham_df  = df_capped[
        df_capped["source"] == "HAM10000"].copy()
    rest_df = df_capped[
        df_capped["source"] != "HAM10000"].copy()

    train_parts, test_parts = [], []

    if len(ham_df) > 0:
        lesions = ham_df.groupby(
            "group_id")["unified_label"].first(
            ).reset_index()
        tr_l, te_l = train_test_split(
            lesions["group_id"].values,
            test_size=0.20,
            random_state=SEED,
            stratify=lesions["unified_label"].values)
        train_parts.append(
            ham_df[ham_df["group_id"].isin(tr_l)])
        test_parts.append(
            ham_df[ham_df["group_id"].isin(te_l)])

    if len(rest_df) > 0:
        tr_r, te_r = train_test_split(
            rest_df,
            test_size=0.20,
            random_state=SEED,
            stratify=rest_df["unified_label"].values)
        train_parts.append(tr_r)
        test_parts.append(te_r)

    test_df = pd.concat(
        test_parts).reset_index(drop=True)
    print(f"  Test split   : {len(test_df):,} images")

    print(f"\n  Test set class distribution:")
    for cls_idx in sorted(
            test_df["unified_label"].unique()):
        name = class_names.get(cls_idx, f"Class {cls_idx}")
        n    = (test_df["unified_label"] == cls_idx).sum()
        print(f"    {cls_idx}: {name:<40} {n:5d}")

    return test_df, class_names, num_classes


# ══════════════════════════════════════════════════════════════════
# STEP 2 — Load model and cached features
# ══════════════════════════════════════════════════════════════════

def load_model_and_cache(num_classes):
    print(f"\n{SEP}")
    print(f"  Step 2 — Loading Model and Feature Cache")
    print(SEP)

    use_agent2   = False
    features_arr = None
    path_to_idx  = {}

    if (os.path.exists(CACHE_FEATURES)
            and os.path.exists(CACHE_INDEX)):
        try:
            cache_idx_df = pd.read_csv(CACHE_INDEX)
            features_arr = np.load(CACHE_FEATURES)
            path_to_idx  = {
                os.path.normpath(p): i
                for i, p in enumerate(
                    cache_idx_df["image_path"])
            }
            if features_arr.shape[1] == AGENT2_DIM:
                use_agent2 = True
                nonzero = (
                    features_arr.sum(axis=1) != 0).sum()
                print(f"  Agent 2 cache : "
                      f"{features_arr.shape} | "
                      f"Non-zero: {nonzero:,}")
            else:
                print(f"  Cache dim mismatch "
                      f"— backbone only")
        except Exception as e:
            print(f"  Cache error: {e} — backbone only")
    else:
        print(f"  No cache found — backbone only")

    combined_dim = (
        BACKBONE_DIM + AGENT2_DIM
        if use_agent2 else BACKBONE_DIM)

    print(f"  Combined dim  : {combined_dim}")
    print(f"  Device        : {DEVICE}")

    model = CombinedModel(
        num_classes  = num_classes,
        use_agent2   = use_agent2,
        combined_dim = combined_dim,
    )

    # Load checkpoint
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    missing, unexpected = model.load_state_dict(
        state, strict=False)

    if missing:
        print(f"  Missing keys  : {missing}")
    if unexpected:
        print(f"  Unexpected    : {unexpected}")

    model = model.to(DEVICE)
    model.eval()
    print(f"  Model loaded  : OK")

    return model, use_agent2, features_arr, path_to_idx


# ══════════════════════════════════════════════════════════════════
# STEP 3 — Run batch inference on full test set
# ══════════════════════════════════════════════════════════════════

def run_inference(model, test_df, use_agent2,
                  features_arr, path_to_idx):
    print(f"\n{SEP}")
    print(f"  Step 3 — Batch Inference")
    print(f"  Test size  : {len(test_df):,} images")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Device     : {DEVICE}")
    print(SEP)

    dataset = EvalDataset(
        test_df, use_agent2, features_arr, path_to_idx)
    loader  = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = DEVICE.type == "cuda",
    )

    all_labels = []
    all_probs  = []
    all_paths  = []

    start = time.time()
    total = len(loader)

    with torch.no_grad():
        for batch_idx, (imgs, a2, labels, paths) in (
                enumerate(loader)):

            imgs   = imgs.to(DEVICE)
            a2_gpu = a2.to(DEVICE) if use_agent2 else None
            logits = model(imgs, a2_gpu)
            probs  = F.softmax(logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

            if (batch_idx + 1) % 10 == 0 or (
                    batch_idx + 1) == total:
                pct     = (batch_idx + 1) / total * 100
                elapsed = time.time() - start
                eta     = (elapsed / (batch_idx + 1) *
                           (total - batch_idx - 1))
                print(f"  Batch {batch_idx+1:4d}/{total}"
                      f"  {pct:5.1f}%"
                      f"  ETA: {eta:5.0f}s")

    elapsed = round(time.time() - start, 1)
    print(f"\n  Inference complete: {elapsed}s")

    return (
        np.array(all_labels),
        np.array(all_probs),
        all_paths,
    )


# ══════════════════════════════════════════════════════════════════
# METRICS HELPER
# ══════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred,
                    class_names, label=""):
    acc  = accuracy_score(y_true, y_pred)
    mf1  = f1_score(y_true, y_pred,
                    average="macro",    zero_division=0)
    wf1  = f1_score(y_true, y_pred,
                    average="weighted", zero_division=0)
    prec = precision_score(
        y_true, y_pred,
        average="macro", zero_division=0)
    rec  = recall_score(
        y_true, y_pred,
        average="macro", zero_division=0)

    class_keys = sorted(class_names.keys())
    pcf1 = f1_score(
        y_true, y_pred,
        average=None,
        zero_division=0,
        labels=class_keys)

    return {
        "label"          : label,
        "n_samples"      : len(y_true),
        "accuracy"       : round(acc  * 100, 2),
        "macro_f1"       : round(mf1  * 100, 2),
        "weighted_f1"    : round(wf1  * 100, 2),
        "macro_precision": round(prec * 100, 2),
        "macro_recall"   : round(rec  * 100, 2),
        "per_class_f1"   : {
            class_names.get(k, str(k)):
            round(pcf1[j] * 100, 2)
            for j, k in enumerate(class_keys)
        },
    }


# ══════════════════════════════════════════════════════════════════
# SYSTEM 1 — Agent 3 alone
# ══════════════════════════════════════════════════════════════════

def evaluate_agent3_alone(y_true, all_probs, class_names):
    """
    System 1: Raw softmax argmax.
    No coordination, no OOD detection.
    This is the baseline every other system is compared against.
    """
    y_pred = np.argmax(all_probs, axis=1)
    m      = compute_metrics(
        y_true, y_pred, class_names,
        label="System 1: Agent 3 Alone")
    m["n_evaluated"] = len(y_true)
    return m


# ══════════════════════════════════════════════════════════════════
# SYSTEM 2 — Agent 3 + OOD Detection
# ══════════════════════════════════════════════════════════════════

def evaluate_with_ood(y_true, all_probs, class_names):
    """
    System 2: Agent 3 + Phase 1 OOD detection.
    Cases where max softmax < OOD_THRESHOLD are flagged
    and excluded from prediction metrics.
    Metrics computed only on in-distribution cases.
    """
    max_confs = np.max(all_probs, axis=1)
    y_pred    = np.argmax(all_probs, axis=1)
    ood_mask  = max_confs < OOD_THRESHOLD

    n_ood     = int(ood_mask.sum())
    n_in_dist = int((~ood_mask).sum())

    print(f"\n  OOD Detection (threshold={OOD_THRESHOLD}):")
    print(f"    Total cases     : {len(y_true):,}")
    print(f"    OOD flagged     : {n_ood:,} "
          f"({n_ood/len(y_true)*100:.2f}%)")
    print(f"    In-distribution : {n_in_dist:,}")

    # Of the flagged OOD cases, how often would
    # Agent 3 have been correct if we had not flagged them?
    if n_ood > 0:
        ood_correct = int(
            (y_pred[ood_mask] == y_true[ood_mask]).sum())
        ood_acc = ood_correct / n_ood * 100
        print(f"    OOD correct if not flagged: "
              f"{ood_correct}/{n_ood} "
              f"({ood_acc:.1f}%) would have been right")
        print(f"    Remaining wrong cases safely "
              f"redirected: "
              f"{n_ood - ood_correct}")

    # Metrics on in-distribution only
    m = compute_metrics(
        y_true[~ood_mask],
        y_pred[~ood_mask],
        class_names,
        label="System 2: Agent 3 + OOD Detection")

    m["n_evaluated"]    = n_in_dist
    m["ood_flagged"]    = n_ood
    m["ood_flagged_pct"]= round(
        n_ood / len(y_true) * 100, 2)
    if n_ood > 0:
        m["ood_would_be_correct_pct"] = round(ood_acc, 2)

    return m, ood_mask


# ══════════════════════════════════════════════════════════════════
# SYSTEM 3 — Full Coordination Pipeline (image only)
# ══════════════════════════════════════════════════════════════════

def evaluate_with_coordination(y_true, all_probs,
                               ood_mask, class_names):
    """
    System 3: Agent 3 + OOD + Coordination Phases 6, 7, 8.
    Image-only mode (no Clinical Agent — no symptoms in test set).

    Decision policy applied:
      OOD_WARNING      : confidence < OOD_THRESHOLD
      ESCALATE         : malignant class predicted
      ROUTINE          : benign, confidence >= 0.70
      CONCORDANT       : benign, confidence 0.50-0.70
      REQUEST_EVIDENCE : benign, confidence 0.40-0.50
    """
    max_confs = np.max(all_probs, axis=1)
    y_pred    = np.argmax(all_probs, axis=1)

    decisions = []
    for i in range(len(y_true)):
        conf   = max_confs[i]
        pred   = int(y_pred[i])
        is_mal = pred in MALIGNANT_CLASSES

        if ood_mask[i]:
            decisions.append("OOD_WARNING")
        elif is_mal:
            decisions.append("ESCALATE")
        elif conf >= 0.70:
            decisions.append("ROUTINE")
        elif conf >= 0.50:
            decisions.append("CONCORDANT")
        else:
            decisions.append("REQUEST_EVIDENCE")

    decisions = np.array(decisions)

    print(f"\n  Coordination Decision Distribution:")
    order = [
        "ROUTINE", "CONCORDANT", "ESCALATE",
        "REQUEST_EVIDENCE", "OOD_WARNING"]
    for d in order:
        n   = int((decisions == d).sum())
        pct = n / len(decisions) * 100
        bar = "=" * int(pct / 2)
        print(f"    {d:<22}: "
              f"{n:5,} ({pct:5.1f}%)  {bar}")

    # Malignant escalation analysis
    n_mal_true = int(sum(
        1 for l in y_true if l in MALIGNANT_CLASSES))
    n_esc      = int((decisions == "ESCALATE").sum())
    print(f"\n  True malignant cases in test set : "
          f"{n_mal_true:,}")
    print(f"  Cases flagged as ESCALATE        : "
          f"{n_esc:,}")
    if n_mal_true > 0:
        esc_rate = n_esc / n_mal_true * 100
        print(f"  Escalation coverage of malignant : "
              f"{esc_rate:.1f}%")

    # Classification metrics on non-OOD cases
    non_ood = ~ood_mask
    m = compute_metrics(
        y_true[non_ood],
        y_pred[non_ood],
        class_names,
        label="System 3: Full Coordination")

    m["n_evaluated"] = int(non_ood.sum())
    m["decision_distribution"] = {
        d: int((decisions == d).sum())
        for d in order
    }
    m["malignant_true"]         = n_mal_true
    m["escalated"]              = n_esc
    if n_mal_true > 0:
        m["escalation_coverage_pct"] = round(
            n_esc / n_mal_true * 100, 2)

    return m, decisions


# ══════════════════════════════════════════════════════════════════
# ABLATION — OOD threshold sensitivity
# Document 1, Step 16 partial
# ══════════════════════════════════════════════════════════════════

def ablation_ood_threshold(y_true, all_probs, class_names):
    """
    Ablation: test multiple OOD thresholds to verify that
    the chosen threshold of 0.40 is optimal.
    Reports macro F1 on the remaining in-distribution cases
    at each threshold.
    """
    print(f"\n  Ablation: OOD Threshold Sensitivity")
    print(f"  {'Threshold':<12} "
          f"{'Flagged':>10} "
          f"{'Remaining':>10} "
          f"{'Macro F1':>10}")
    print(f"  {'-'*45}")

    y_pred    = np.argmax(all_probs, axis=1)
    max_confs = np.max(all_probs, axis=1)
    results   = {}

    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        in_dist   = max_confs >= thresh
        n_flagged = int((~in_dist).sum())
        n_remain  = int(in_dist.sum())
        if n_remain < 10:
            continue
        mf1 = f1_score(
            y_true[in_dist],
            y_pred[in_dist],
            average="macro",
            zero_division=0)
        marker = " <-- selected" if thresh == OOD_THRESHOLD else ""
        print(f"  {thresh:<12.2f} "
              f"{n_flagged:>10,} "
              f"{n_remain:>10,} "
              f"{mf1*100:>10.2f}%"
              f"{marker}")
        results[str(thresh)] = {
            "n_flagged": n_flagged,
            "n_remaining": n_remain,
            "macro_f1": round(mf1 * 100, 2),
        }

    return results


# ══════════════════════════════════════════════════════════════════
# PRINT COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════

def print_comparison_table(m1, m2, m3, class_names):
    """
    Print the final comparison table for all three systems.
    This is Table 1 in the paper.
    """
    print(f"\n\n{SEP}")
    print(f"  FINAL RESULTS — BASELINE COMPARISON")
    print(f"  MAAI-SkinDx | Document 1, Step 1")
    print(SEP)

    # Overall metrics
    print(f"\n  {'System':<38} "
          f"{'N':>6} "
          f"{'Acc%':>6} "
          f"{'MacF1%':>7} "
          f"{'WtF1%':>6} "
          f"{'Prec%':>6} "
          f"{'Rec%':>6}")
    print(f"  {SEP2}")

    for m in [m1, m2, m3]:
        n = m.get("n_evaluated", m.get("n_samples", 0))
        print(
            f"  {m['label']:<38} "
            f"{n:>6,} "
            f"{m['accuracy']:>6.2f} "
            f"{m['macro_f1']:>7.2f} "
            f"{m['weighted_f1']:>6.2f} "
            f"{m['macro_precision']:>6.2f} "
            f"{m['macro_recall']:>6.2f}"
        )

    # Per-class F1 comparison
    print(f"\n  Per-Class F1 Comparison:")
    print(f"  {'Class':<38} "
          f"{'S1 Alone':>9} "
          f"{'S2+OOD':>9} "
          f"{'S3+Coord':>9} "
          f"{'Delta':>7}")
    print(f"  {SEP2}")

    r1 = m1["per_class_f1"]
    r2 = m2["per_class_f1"]
    r3 = m3["per_class_f1"]

    for cls in r1:
        v1    = r1.get(cls, 0)
        v2    = r2.get(cls, 0)
        v3    = r3.get(cls, 0)
        delta = v3 - v1
        sign  = "+" if delta >= 0 else ""
        print(
            f"  {cls:<38} "
            f"{v1:>9.2f} "
            f"{v2:>9.2f} "
            f"{v3:>9.2f} "
            f"{sign}{delta:>6.2f}"
        )

    # Key takeaways
    print(f"\n  Key Takeaways:")
    delta_f1 = m3["macro_f1"] - m1["macro_f1"]
    sign     = "+" if delta_f1 >= 0 else ""
    print(f"    Macro F1 change (S1 to S3) : "
          f"{sign}{delta_f1:.2f}%")
    print(f"    OOD cases safely flagged   : "
          f"{m2.get('ood_flagged', 0):,} "
          f"({m2.get('ood_flagged_pct', 0):.2f}%)")
    print(f"    Malignant cases escalated  : "
          f"{m3.get('escalated', 0):,} / "
          f"{m3.get('malignant_true', 0):,} "
          f"({m3.get('escalation_coverage_pct', 0):.1f}%)")
    print(SEP)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    wall_start = time.time()

    print(f"\n{SEP}")
    print(f"  MAAI-SkinDx | Baseline Evaluation")
    print(f"  Document 1 — Step 1")
    print(f"  Date: "
          f"{datetime.now().strftime('%d %B %Y %H:%M')}")
    print(f"  Device: {DEVICE}")
    print(SEP)

    # Step 1 — Reproduce test split
    test_df, class_names, num_classes = (
        reproduce_test_split())

    # Step 2 — Load model
    model, use_agent2, features_arr, path_to_idx = (
        load_model_and_cache(num_classes))

    # Step 3 — Run inference
    y_true, all_probs, all_paths = run_inference(
        model, test_df, use_agent2,
        features_arr, path_to_idx)

    # Inference summary
    confs = np.max(all_probs, axis=1)
    print(f"\n  Inference summary:")
    print(f"    Total images     : {len(y_true):,}")
    print(f"    Mean confidence  : "
          f"{confs.mean()*100:.1f}%")
    print(f"    Min confidence   : "
          f"{confs.min()*100:.1f}%")
    print(f"    Max confidence   : "
          f"{confs.max()*100:.1f}%")
    print(f"    Below OOD (0.40) : "
          f"{(confs < OOD_THRESHOLD).sum():,}")

    # System 1 — Agent 3 alone
    print(f"\n{SEP}")
    print(f"  System 1 — Agent 3 Alone")
    print(SEP)
    m1 = evaluate_agent3_alone(
        y_true, all_probs, class_names)
    print(f"  Accuracy   : {m1['accuracy']}%")
    print(f"  Macro F1   : {m1['macro_f1']}%")
    print(f"  Weighted F1: {m1['weighted_f1']}%")

    # System 2 — Agent 3 + OOD
    print(f"\n{SEP}")
    print(f"  System 2 — Agent 3 + OOD Detection")
    print(SEP)
    m2, ood_mask = evaluate_with_ood(
        y_true, all_probs, class_names)
    print(f"  Accuracy   : {m2['accuracy']}% "
          f"(on {m2['n_evaluated']:,} cases)")
    print(f"  Macro F1   : {m2['macro_f1']}%")
    print(f"  OOD flagged: {m2['ood_flagged']:,}")

    # System 3 — Full coordination
    print(f"\n{SEP}")
    print(f"  System 3 — Full Coordination Pipeline")
    print(SEP)
    m3, decisions = evaluate_with_coordination(
        y_true, all_probs, ood_mask, class_names)
    print(f"  Accuracy   : {m3['accuracy']}%")
    print(f"  Macro F1   : {m3['macro_f1']}%")

    # Ablation
    print(f"\n{SEP}")
    print(f"  Ablation — OOD Threshold Sensitivity")
    print(SEP)
    ablation = ablation_ood_threshold(
        y_true, all_probs, class_names)

    # Comparison table
    print_comparison_table(m1, m2, m3, class_names)

    # Save results
    wall_time = round(time.time() - wall_start, 1)
    output    = {
        "date"         : datetime.now().isoformat(),
        "seed"         : SEED,
        "test_size"    : int(len(y_true)),
        "ood_threshold": OOD_THRESHOLD,
        "system_1"     : m1,
        "system_2"     : m2,
        "system_3"     : m3,
        "ablation"     : ablation,
        "elapsed_s"    : wall_time,
    }

    save_path = os.path.join(
        OUTPUT_DIR, "baseline_comparison.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved : {save_path}")
    print(f"  Total time    : {wall_time}s")
    print(f"{SEP}\n")

    return output


if __name__ == "__main__":
    main()