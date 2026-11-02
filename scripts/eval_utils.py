# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Evaluation Utilities
# Shared functions for all evaluation scripts.
# Reproduces the exact test split from training (SEED=42).
# ══════════════════════════════════════════════════════════════════

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

from PIL import Image as PILImage
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))

# ── Config — must match training exactly ──────────────────────────
SEED              = 42
IMG_SIZE          = 224
BACKBONE_DIM      = 2048
AGENT2_DIM        = 6916
MAX_PER_CLASS     = 6000
MAX_HAM_PER_CLASS = 2000
MALIGNANT_CLASSES = {0, 2, 3}
OOD_THRESHOLD     = 0.40
BATCH_SIZE        = 64

UNIFIED_CSV   = os.path.join(
    PROJECT_ROOT, "data", "processed",
    "agent3_unified_dataset.csv")
LABEL_MAP_CSV = os.path.join(
    PROJECT_ROOT, "data", "processed",
    "agent3_label_mapping.csv")
CACHE_FEATURES = os.path.join(
    PROJECT_ROOT, "data", "processed",
    "agent2_cache_features.npy")
CACHE_INDEX    = os.path.join(
    PROJECT_ROOT, "data", "processed",
    "agent2_cache_index.csv")
MODEL_PATH    = os.path.join(
    PROJECT_ROOT, "models", "agent3",
    "agent3_resnet50_final.pth")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")


# ── Model ──────────────────────────────────────────────────────────

class CombinedModel(nn.Module):
    def __init__(self, num_classes, use_agent2, combined_dim):
        super().__init__()
        base = models.resnet50(weights=None)
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool
        self.use_agent2 = use_agent2
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
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


# ── Dataset ────────────────────────────────────────────────────────

class EvalDataset(Dataset):
    def __init__(self, df, use_agent2, features_arr, path_to_idx):
        self.paths      = [os.path.normpath(p)
                           for p in df["image_path"]]
        self.labels     = [int(l) for l in df["unified_label"]]
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
            a2   = (torch.from_numpy(
                        self.features[cidx].copy()).float()
                    if cidx >= 0
                    else torch.zeros(AGENT2_DIM))
        else:
            a2 = torch.zeros(AGENT2_DIM)
        return img, a2, label, path


# ── Reproduce test split — identical to training ───────────────────

def reproduce_test_split():
    """
    Reproduce the exact test split used during Agent 3 training.
    Returns test_df and class_names dict.
    """
    df = pd.read_csv(UNIFIED_CSV)
    df["exists"] = df["image_path"].apply(os.path.exists)
    df = df[df["exists"]].copy().reset_index(drop=True)
    df["image_path"] = df["image_path"].apply(os.path.normpath)

    label_map   = pd.read_csv(LABEL_MAP_CSV)
    class_names = {
        int(r["class_index"]): r["class_name"]
        for _, r in label_map.iterrows()
        if int(r["class_index"]) in df["unified_label"].unique()
    }

    # Class capping
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
        capped_rows.append(pd.concat([ham_rows, rest_rows]))

    df_capped = pd.concat(capped_rows).reset_index(drop=True)

    # Train/test split
    ham_df  = df_capped[df_capped["source"] == "HAM10000"].copy()
    rest_df = df_capped[df_capped["source"] != "HAM10000"].copy()

    train_parts, test_parts = [], []

    if len(ham_df) > 0:
        lesions = ham_df.groupby(
            "group_id")["unified_label"].first().reset_index()
        tr_l, te_l = train_test_split(
            lesions["group_id"].values,
            test_size=0.20, random_state=SEED,
            stratify=lesions["unified_label"].values)
        train_parts.append(ham_df[ham_df["group_id"].isin(tr_l)])
        test_parts.append(ham_df[ham_df["group_id"].isin(te_l)])

    if len(rest_df) > 0:
        tr_r, te_r = train_test_split(
            rest_df, test_size=0.20, random_state=SEED,
            stratify=rest_df["unified_label"].values)
        train_parts.append(tr_r)
        test_parts.append(te_r)

    test_df = pd.concat(test_parts).reset_index(drop=True)
    return test_df, class_names


# ── Load model ─────────────────────────────────────────────────────

def load_agent3_model():
    """Load the Agent 3 model from checkpoint."""
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    use_agent2  = ckpt.get("use_agent2", True)
    combined_dim = ckpt.get("combined_dim",
                             BACKBONE_DIM + AGENT2_DIM)
    num_classes  = ckpt.get("num_classes", 9)

    model = CombinedModel(num_classes, use_agent2, combined_dim)
    state_key = "model_state_dict" if "model_state_dict" in ckpt else "state_dict"
    model.load_state_dict(ckpt[state_key])
    model = model.to(DEVICE)
    model.eval()
    print(f"  Agent 3 loaded — {num_classes} classes  "
          f"use_agent2={use_agent2}  dim={combined_dim}")
    return model, use_agent2


# ── Load Agent 2 features ──────────────────────────────────────────

def load_agent2_features():
    """Load cached Agent 2 features."""
    if not os.path.exists(CACHE_FEATURES):
        print("  Agent 2 cache not found — image-only mode")
        return None, {}
    features_arr = np.load(CACHE_FEATURES)
    cache_idx    = pd.read_csv(CACHE_INDEX)
    path_to_idx  = {
        os.path.normpath(row["image_path"]): idx
        for idx, row in cache_idx.iterrows()
    }
    print(f"  Agent 2 cache loaded — {len(features_arr):,} entries")
    return features_arr, path_to_idx


# ── Run inference ──────────────────────────────────────────────────

def run_inference(model, test_df, use_agent2,
                  features_arr, path_to_idx):
    """
    Run Agent 3 inference on the test set.
    Returns arrays: preds, confs, probs, labels.
    """
    dataset = EvalDataset(
        test_df, use_agent2, features_arr, path_to_idx)
    loader  = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True)

    all_preds  = []
    all_confs  = []
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for imgs, a2_feats, labels, _ in loader:
            imgs     = imgs.to(DEVICE)
            a2_feats = a2_feats.to(DEVICE)
            out      = model(imgs,
                             a2_feats if use_agent2 else None)
            probs    = torch.softmax(out, dim=1).cpu().numpy()
            preds    = probs.argmax(axis=1)
            confs    = probs.max(axis=1)
            all_preds.extend(preds.tolist())
            all_confs.extend(confs.tolist())
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

    return (np.array(all_preds),
            np.array(all_confs),
            np.array(all_probs),
            np.array(all_labels))


# ── Full test data loader ──────────────────────────────────────────

def load_test_predictions(verbose=True):
    """
    Full pipeline: load split, model, features, run inference.
    Returns preds, confs, probs, labels, class_names.
    """
    if verbose:
        print("  Reproducing test split ...")
    test_df, class_names = reproduce_test_split()
    if verbose:
        print(f"  Test set: {len(test_df):,} images  "
              f"{len(class_names)} classes")

    model, use_agent2 = load_agent3_model()
    features_arr, path_to_idx = load_agent2_features()

    if verbose:
        print(f"  Running inference ...")
    preds, confs, probs, labels = run_inference(
        model, test_df, use_agent2,
        features_arr, path_to_idx)

    if verbose:
        in_dist = (confs >= OOD_THRESHOLD).sum()
        print(f"  Inference done — "
              f"total={len(preds):,}  "
              f"in-dist={in_dist:,}  "
              f"OOD={(~(confs >= OOD_THRESHOLD)).sum():,}")

    return preds, confs, probs, labels, class_names
