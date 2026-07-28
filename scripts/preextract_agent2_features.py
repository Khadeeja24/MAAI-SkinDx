# MAAI-SkinDx | Agent 2 Feature Pre-Extraction (Incremental)
# Loads old cache, only extracts NEW images not in old cache.
# Uses NO CAP on malignant classes — same as train script.
# Time: ~2-4 hours instead of 48 hours.

import os, sys, time, random
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
import io, contextlib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

UNIFIED_CSV    = os.path.join(PROJECT_ROOT, "data", "processed", "agent3_unified_dataset.csv")
CACHE_FEATURES = os.path.join(PROJECT_ROOT, "data", "processed", "agent2_cache_features.npy")
CACHE_INDEX    = os.path.join(PROJECT_ROOT, "data", "processed", "agent2_cache_index.csv")

SEED              = 42
MAX_PER_CLASS     = 6000
MAX_HAM_PER_CLASS = 2000
MALIGNANT_CLASSES = {0, 2, 3}
AGENT2_DIM        = 6916
CHECKPOINT_EVERY  = 200

random.seed(SEED)
np.random.seed(SEED)

if not os.path.exists(UNIFIED_CSV):
    print("ERROR: Run prepare_agent3_data.py first")
    sys.exit(1)

print(f"\n{'='*65}")
print(f"  MAAI-SkinDx | Agent 2 Feature Pre-Extraction (Incremental)")
print(f"{'='*65}\n")

# Step 1: Compute capped set (identical to train script)
print("── Step 1: Computing capped image set ──")
df = pd.read_csv(UNIFIED_CSV)
df["exists"] = df["image_path"].apply(os.path.exists)
df = df[df["exists"]].copy().reset_index(drop=True)
print(f"  Valid images: {len(df):,}")

capped_rows = []
for cls_idx in sorted(df["unified_label"].unique()):
    cls_df    = df[df["unified_label"] == cls_idx].copy()
    ham_rows  = cls_df[cls_df["source"] == "HAM10000"]
    rest_rows = cls_df[cls_df["source"] != "HAM10000"]

    if cls_idx in MALIGNANT_CLASSES:
        capped_rows.append(cls_df)
        print(f"  Class {cls_idx}: {len(cls_df):,} — NO CAP (malignant)")
        continue

    if len(ham_rows) > MAX_HAM_PER_CLASS:
        ham_rows = ham_rows.sample(n=MAX_HAM_PER_CLASS, random_state=SEED)

    remaining = MAX_PER_CLASS - len(ham_rows)
    if len(rest_rows) > remaining:
        rest_rows = rest_rows.sample(n=remaining, random_state=SEED)

    capped_rows.append(pd.concat([ham_rows, rest_rows]))

capped_df = pd.concat(capped_rows).reset_index(drop=True)
all_paths = [os.path.normpath(p) for p in capped_df["image_path"].tolist()]
N = len(all_paths)
print(f"  Total capped: {N:,}")

# Step 2: Load old cache
print(f"\n── Step 2: Loading Old Cache ──")
old_cache = {}
if os.path.exists(CACHE_INDEX) and os.path.exists(CACHE_FEATURES):
    try:
        old_idx_df   = pd.read_csv(CACHE_INDEX)
        old_features = np.load(CACHE_FEATURES)
        for i, p in enumerate(old_idx_df["image_path"]):
            norm_p = os.path.normpath(str(p))
            if i < len(old_features):
                old_cache[norm_p] = old_features[i]
        print(f"  Old cache: {len(old_cache):,} entries")
    except Exception as e:
        print(f"  Cache error: {e} — extracting everything")
else:
    print("  No old cache — extracting everything")

# Step 3: Check coverage
cached_paths = [p for p in all_paths if p in old_cache]
new_paths    = [p for p in all_paths if p not in old_cache]
print(f"\n── Step 3: Coverage ──")
print(f"  Already cached : {len(cached_paths):,} ({len(cached_paths)/N*100:.1f}%)")
print(f"  Need extraction: {len(new_paths):,} ({len(new_paths)/N*100:.1f}%)")
print(f"  Estimated time : ~{len(new_paths)*0.8/3600:.1f} hours")

# Step 4: Build array from old cache
features_arr = np.zeros((N, AGENT2_DIM), dtype=np.float32)
for i, path in enumerate(all_paths):
    if path in old_cache:
        features_arr[i] = old_cache[path]
print(f"\n  Filled from cache: {len(cached_paths):,}")

# Step 5: Extract new images
if len(new_paths) > 0:
    print(f"\n── Step 4: Extracting {len(new_paths):,} New Images ──")
    from agents.agent2_feature_extraction import FeatureExtractionAgent
    agent2 = FeatureExtractionAgent()
    print("  Agent 2 ready\n")

    new_path_set = set(new_paths)
    failed = 0; done = 0
    start_t = time.time()

    for i, path in enumerate(all_paths):
        if path not in new_path_set:
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                result = agent2.run(path)
            if result.get("status") == "PASS" and result.get("feature_vector") is not None:
                feat = np.asarray(result["feature_vector"], dtype=np.float32)
                if feat.shape[0] == AGENT2_DIM:
                    features_arr[i] = feat
                else:
                    failed += 1
            else:
                failed += 1
        except Exception:
            failed += 1

        done += 1
        elapsed   = time.time() - start_t
        rate      = done / max(elapsed, 1)
        remaining = (len(new_paths) - done) / max(rate, 0.001)

        if done % CHECKPOINT_EVERY == 0 or done == len(new_paths):
            np.save(CACHE_FEATURES, features_arr)
            pd.DataFrame({"image_path": all_paths}).to_csv(CACHE_INDEX, index=False)
            pct = done / len(new_paths) * 100
            print(f"  [{done:5,}/{len(new_paths):,}] {pct:.1f}% | "
                  f"Elapsed: {elapsed/3600:.2f}h | "
                  f"ETA: ~{remaining/3600:.2f}h | "
                  f"Failed: {failed}")
else:
    print(f"\n── Step 4: Nothing to extract — all cached ──")

np.save(CACHE_FEATURES, features_arr)
pd.DataFrame({"image_path": all_paths}).to_csv(CACHE_INDEX, index=False)

nonzero = (features_arr.sum(axis=1) != 0).sum()
print(f"\n{'='*65}")
print(f"  COMPLETE")
print(f"  Total    : {N:,} | Non-zero: {nonzero:,} ({nonzero/N*100:.1f}%)")
print(f"  Shape    : {features_arr.shape}")
print(f"  Next     : python scripts/train_agent3_final.py")
print(f"{'='*65}\n")