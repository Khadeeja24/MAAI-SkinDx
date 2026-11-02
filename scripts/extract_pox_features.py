# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Pox Classifier — Feature Extraction
# Uses DINOv2 (768-dim) + ABCD (4-dim) = 772-dim
# Derm Foundation excluded — pox diseases are visually distinct
# DINOv2 batch 32 on GPU — completes in under 10 minutes
# ══════════════════════════════════════════════════════════════════

import os, sys, time, warnings
import numpy as np
import pandas as pd
import cv2
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import timm
from torchvision import transforms
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────
POX_DIR    = os.path.join(PROJECT_ROOT, "data", "raw", "pox_dataset")
OUT_FEAT   = os.path.join(PROJECT_ROOT, "data", "processed", "pox_features.npy")
OUT_LABELS = os.path.join(PROJECT_ROOT, "data", "processed", "pox_labels.npy")
OUT_INDEX  = os.path.join(PROJECT_ROOT, "data", "processed", "pox_index.csv")

# ── Config ────────────────────────────────────────────────────────
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE     = 224
DINOV2_DIM     = 768
ABCD_DIM       = 4
TOTAL_DIM      = 772
BATCH_SIZE     = 32
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

POX_CLASSES = {
    "Monkeypox" : 0,
    "Chickenpox": 1,
    "Measles"   : 2,
    "Cowpox"    : 3,
    "HFMD"      : 4,
}

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

def l2_norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def minmax_l2_norm(v):
    mn, mx = v.min(), v.max()
    v = (v - mn) / (mx - mn) if mx > mn else v
    return l2_norm(v)

print(f"\n{'='*60}")
print(f"  MAAI-SkinDx | Pox Feature Extraction")
print(f"  Features : DINOv2 768-dim + ABCD 4-dim = 772-dim")
print(f"  Device   : {DEVICE}")
print(f"  Batch    : {BATCH_SIZE}")
print(f"{'='*60}\n")

# ── Step 1: Collect image paths ───────────────────────────────────
print("── Step 1: Collecting images ──")
all_paths, all_labels = [], []

for cls, idx in POX_CLASSES.items():
    d = os.path.join(POX_DIR, cls)
    if not os.path.exists(d):
        print(f"  MISSING: {cls}")
        continue
    imgs = sorted([
        os.path.normpath(os.path.join(d, f))
        for f in os.listdir(d)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    all_paths.extend(imgs)
    all_labels.extend([idx] * len(imgs))
    print(f"  {cls:<15}: {len(imgs):,}")

N = len(all_paths)
print(f"  Total    : {N:,}\n")

# ── Step 2: Load DINOv2 ───────────────────────────────────────────
print("── Step 2: Loading DINOv2 ──")
model = timm.create_model(
    "vit_base_patch14_dinov2.lvd142m",
    pretrained=True,
    num_classes=0,
    img_size=IMAGE_SIZE,
)
for p in model.parameters():
    p.requires_grad = False
model.eval().to(DEVICE)
print(f"  DINOv2 ready on {DEVICE}\n")

# ── Step 3: DINOv2 batch inference ────────────────────────────────
print(f"── Step 3: DINOv2 batch inference ({N:,} images) ──")
dino_feats = np.zeros((N, DINOV2_DIM), dtype=np.float32)
t0 = time.time()

for bs in range(0, N, BATCH_SIZE):
    batch_paths = all_paths[bs: bs + BATCH_SIZE]
    tensors, valid_idx = [], []

    for j, path in enumerate(batch_paths):
        try:
            img = cv2.imread(path)
            if img is None:
                raise ValueError()
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensors.append(TRANSFORM(Image.fromarray(rgb)))
            valid_idx.append(bs + j)
        except Exception:
            pass

    if not tensors:
        continue

    with torch.no_grad():
        emb = model(torch.stack(tensors).to(DEVICE))
    emb = emb.cpu().numpy()

    for k, idx in enumerate(valid_idx):
        dino_feats[idx] = emb[k]

    done = bs + len(batch_paths)
    if done % 1000 == 0 or done >= N:
        elapsed = time.time() - t0
        eta     = elapsed / done * (N - done) if done < N else 0
        print(f"  [{done:,}/{N:,}] "
              f"Elapsed: {elapsed/60:.1f} min | "
              f"ETA: {eta/60:.1f} min")

print(f"  DINOv2 done in {round(time.time()-t0, 1)}s\n")

del model
torch.cuda.empty_cache()

# ── Step 4: ABCD scores + combine ────────────────────────────────
print("── Step 4: ABCD scores + combining features ──")
from agents.agent2_feature_extraction.streams.abcd_scores \
    import compute_abcd_scores

final_feats = np.zeros((N, TOTAL_DIM), dtype=np.float32)
failed = 0
t0 = time.time()

for i, path in enumerate(all_paths):
    try:
        img = cv2.imread(path)
        if img is None:
            raise ValueError()
        abcd_raw = compute_abcd_scores(img)
    except Exception:
        abcd_raw = np.zeros(ABCD_DIM, dtype=np.float32)
        failed  += 1

    dino_n = l2_norm(dino_feats[i])
    abcd_n = minmax_l2_norm(abcd_raw)

    final_feats[i] = np.concatenate([dino_n, abcd_n])

    if (i + 1) % 1000 == 0 or (i + 1) == N:
        print(f"  [{i+1:,}/{N:,}] ABCD done")

print(f"  Combine done in {round(time.time()-t0,1)}s\n")

# ── Step 5: Save ──────────────────────────────────────────────────
np.save(OUT_FEAT,   final_feats)
np.save(OUT_LABELS, np.array(all_labels, dtype=np.int32))
pd.DataFrame({
    "image_path": all_paths,
    "label"     : all_labels,
}).to_csv(OUT_INDEX, index=False)

nonzero = int((final_feats.sum(axis=1) != 0).sum())

print(f"{'='*60}")
print(f"  COMPLETE")
print(f"  Shape    : {final_feats.shape}")
print(f"  Extracted: {nonzero:,} / {N:,}")
print(f"  Failed   : {failed:,}")
print(f"\n  Class breakdown:")
labels_arr = np.array(all_labels)
for name, idx in POX_CLASSES.items():
    mask = labels_arr == idx
    ext  = int((final_feats[mask].sum(axis=1) != 0).sum())
    print(f"    {idx} {name:<15}: "
          f"{mask.sum():,} total | {ext:,} extracted")
print(f"\n  Saved:")
print(f"    {OUT_FEAT}")
print(f"    {OUT_LABELS}")
print(f"    {OUT_INDEX}")
print(f"\n  Next: python scripts/train_pox_mlp.py")
print(f"{'='*60}\n")