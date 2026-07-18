# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 3 — Data Preparation Script
# ══════════════════════════════════════════════════════════════════
# Run this ONCE before training.
# Loads all 4 datasets, maps every label to unified 9-class
# taxonomy, validates every image exists on disk, saves one
# clean unified CSV ready for training.
#
# Output: data/processed/agent3_unified_dataset.csv
#
# Verified dataset structures:
#
# HAM10000:
#   data/raw/ham10000/HAM10000_images_part_1/*.jpg
#   data/raw/ham10000/HAM10000_images_part_2/*.jpg
#   data/raw/ham10000/HAM10000_metadata (CSV, label col=dx)
#
# Fitzpatrick-17k:
#   data/raw/fitzpatrick17k/fitzpatrick_3groups/group1_light/*.jpg
#   data/raw/fitzpatrick17k/fitzpatrick_3groups/group2_medium/*.jpg
#   data/raw/fitzpatrick17k/fitzpatrick_3groups/group3_dark/*.jpg
#   data/raw/fitzpatrick17k/fitzpatrick17k.csv (label col=label)
#
# PAD-UFES-20:
#   data/raw/pad_ufes20/imgs_part_1/*.png
#   data/raw/pad_ufes20/imgs_part_2/*.png
#   data/raw/pad_ufes20/imgs_part_3/*.png
#   data/raw/pad_ufes20/metadata.csv (label col=diagnostic,
#                                      image col=img_id)
#
# Massive Balanced:
#   data/raw/massive_balanced/balanced_dataset/<category>/*.jpg
#   Folder name IS the class label
#
# DDI-31 (TEST ONLY — never used in training):
#   data/raw/ddi31/*.png
#   data/raw/ddi31/ddi_metadata.csv (label col=disease,
#                                     image col=DDI_file)
#
# Unified 9-class taxonomy:
#   0: Melanoma
#   1: Melanocytic Nevus
#   2: Basal Cell Carcinoma
#   3: Actinic Keratosis / Squamous Cell Carcinoma
#   4: Benign Keratosis / Seborrheic Keratosis
#   5: Vascular Lesion
#   6: Dermatofibroma
#   7: Inflammatory (psoriasis, lichen planus, lupus, etc.)
#   8: Eczema / Dermatitis
# ══════════════════════════════════════════════════════════════════

import os
import sys
import pandas as pd
from pathlib import Path
from collections import Counter

# ─── Project root ──────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Dataset paths ─────────────────────────────────────────────────
HAM_META    = os.path.join(PROJECT_ROOT, "data", "raw", "ham10000", "HAM10000_metadata")
HAM_PART1   = os.path.join(PROJECT_ROOT, "data", "raw", "ham10000", "HAM10000_images_part_1")
HAM_PART2   = os.path.join(PROJECT_ROOT, "data", "raw", "ham10000", "HAM10000_images_part_2")

FITZ_CSV    = os.path.join(PROJECT_ROOT, "data", "raw", "fitzpatrick17k", "fitzpatrick17k.csv")
FITZ_G1     = os.path.join(PROJECT_ROOT, "data", "raw", "fitzpatrick17k", "fitzpatrick_3groups", "group1_light")
FITZ_G2     = os.path.join(PROJECT_ROOT, "data", "raw", "fitzpatrick17k", "fitzpatrick_3groups", "group2_medium")
FITZ_G3     = os.path.join(PROJECT_ROOT, "data", "raw", "fitzpatrick17k", "fitzpatrick_3groups", "group3_dark")

PAD_CSV     = os.path.join(PROJECT_ROOT, "data", "raw", "pad_ufes20", "metadata.csv")
PAD_PART1   = os.path.join(PROJECT_ROOT, "data", "raw", "pad_ufes20", "imgs_part_1")
PAD_PART2   = os.path.join(PROJECT_ROOT, "data", "raw", "pad_ufes20", "imgs_part_2")
PAD_PART3   = os.path.join(PROJECT_ROOT, "data", "raw", "pad_ufes20", "imgs_part_3")

MASSIVE_ROOT = os.path.join(PROJECT_ROOT, "data", "raw", "massive_balanced", "balanced_dataset")

DDI_CSV     = os.path.join(PROJECT_ROOT, "data", "raw", "ddi31", "ddi_metadata.csv")
DDI_IMGS    = os.path.join(PROJECT_ROOT, "data", "raw", "ddi31")

# ─── Output ────────────────────────────────────────────────────────
OUTPUT_DIR      = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_CSV      = os.path.join(OUTPUT_DIR, "agent3_unified_dataset.csv")
OUTPUT_DDI_CSV  = os.path.join(OUTPUT_DIR, "agent3_ddi31_test.csv")
OUTPUT_LABEL_MAP= os.path.join(OUTPUT_DIR, "agent3_label_mapping.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# SECTION 1 — Unified 9-class taxonomy
# ══════════════════════════════════════════════════════════════════

UNIFIED_CLASSES = {
    0: "Melanoma",
    1: "Melanocytic Nevus",
    2: "Basal Cell Carcinoma",
    3: "Actinic Keratosis / Squamous Cell Carcinoma",
    4: "Benign Keratosis",
    5: "Vascular Lesion",
    6: "Dermatofibroma",
    7: "Inflammatory",
    8: "Eczema / Dermatitis",
}

# ── HAM10000: dx column → unified label ───────────────────────────
HAM_MAP = {
    "mel"  : 0,
    "nv"   : 1,
    "bcc"  : 2,
    "akiec": 3,
    "bkl"  : 4,
    "vasc" : 5,
    "df"   : 6,
}

# ── Fitzpatrick-17k: label column (lowercase) → unified label ─────
FITZ_MAP = {
    # Melanoma
    "melanoma"                       : 0,
    "nodular melanoma"               : 0,
    "lentigo maligna"                : 0,
    "lentigo maligna melanoma"       : 0,
    "acral lentiginous melanoma"     : 0,
    "amelanotic melanoma"            : 0,
    "melanoma in situ"               : 0,
    "superficial spreading melanoma" : 0,
    "desmoplastic melanoma"          : 0,
    # Nevus
    "melanocytic nevi"               : 1,
    "blue nevus"                     : 1,
    "compound nevus"                 : 1,
    "intradermal nevus"              : 1,
    "junctional nevus"               : 1,
    "congenital nevus"               : 1,
    "dysplastic nevus"               : 1,
    "spitz nevus"                    : 1,
    "halo nevus"                     : 1,
    "becker nevus"                   : 1,
    "nevus"                          : 1,
    # BCC
    "basal cell carcinoma"           : 2,
    # AK / SCC
    "squamous cell carcinoma"        : 3,
    "actinic keratosis"              : 3,
    "bowen's disease"                : 3,
    "keratoacanthoma"                : 3,
    "cutaneous horn"                 : 3,
    # Benign keratosis
    "seborrheic keratosis"           : 4,
    "solar lentigo"                  : 4,
    "dermatosis papulosa nigra"      : 4,
    "stucco keratosis"               : 4,
    "lichenoid keratosis"            : 4,
    # Vascular
    "hemangioma"                     : 5,
    "pyogenic granuloma"             : 5,
    "cherry angioma"                 : 5,
    "venous lake"                    : 5,
    "angiokeratoma"                  : 5,
    "angioma"                        : 5,
    "lymphangioma"                   : 5,
    "port wine stain"                : 5,
    # Dermatofibroma
    "dermatofibroma"                 : 6,
    # Inflammatory
    "psoriasis"                      : 7,
    "lichen planus"                  : 7,
    "lupus erythematosus"            : 7,
    "sarcoidosis"                    : 7,
    "neutrophilic dermatoses"        : 7,
    "photodermatoses"                : 7,
    "pityriasis rosea"               : 7,
    "granuloma annulare"             : 7,
    "morphea"                        : 7,
    "vitiligo"                       : 7,
    "urticaria"                      : 7,
    "drug eruptions"                 : 7,
    "drug eruption"                  : 7,
    "lichen sclerosus"               : 7,
    "rosacea"                        : 7,
    "acne"                           : 7,
    "pityriasis lichenoides"         : 7,
    "folliculitis"                   : 7,
    "hidradenitis suppurativa"       : 7,
    "prurigo nodularis"              : 7,
    "erythema multiforme"            : 7,
    "erythema nodosum"               : 7,
    # Eczema
    "eczema"                         : 8,
    "atopic dermatitis"              : 8,
    "allergic contact dermatitis"    : 8,
    "contact dermatitis"             : 8,
    "irritant contact dermatitis"    : 8,
    "seborrheic dermatitis"          : 8,
    "nummular eczema"                : 8,
    "dyshidrotic eczema"             : 8,
    "stasis dermatitis"              : 8,
    "perioral dermatitis"            : 8,
}

# ── PAD-UFES-20: diagnostic column → unified label ────────────────
PAD_MAP = {
    "BCC": 2,
    "SCC": 3,
    "ACK": 3,
    "SEK": 4,
    "BOD": 3,
    "NEV": 1,
}

# ── Massive Balanced: EXACT folder names → unified label ──────────
# None = skip (not relevant to skin disease diagnosis)
MASSIVE_MAP = {
    "Acne And Rosacea Photos"                                           : 7,
    "Actinic Keratosis Basal Cell Carcinoma And Other Malignant Lesions": 3,
    "Atopic Dermatitis Photos"                                          : 8,
    "Ba  Cellulitis"                                                    : None,
    "Ba Impetigo"                                                       : None,
    "Benign"                                                            : 4,
    "Bullous Disease Photos"                                            : 7,
    "Cellulitis Impetigo And Other Bacterial Infections"                : None,
    "Eczema Photos"                                                     : 8,
    "Exanthems And Drug Eruptions"                                      : 7,
    "Fu Athlete Foot"                                                   : None,
    "Fu Nail Fungus"                                                    : None,
    "Fu Ringworm"                                                       : None,
    "Hair Loss Photos Alopecia And Other Hair Diseases"                 : None,
    "Heathy"                                                            : None,
    "Herpes Hpv And Other Stds Photos"                                  : None,
    "Light Diseases And Disorders Of Pigmentation"                      : 7,
    "Lupus And Other Connective Tissue Diseases"                        : 7,
    "Malignant"                                                         : 0,
    "Melanoma Skin Cancer Nevi And Moles"                               : None,
    "Nail Fungus And Other Nail Disease"                                : None,
    "Pa Cutaneous Larva Migrans"                                        : None,
    "Poison Ivy Photos And Other Contact Dermatitis"                    : 8,
    "Psoriasis Pictures Lichen Planus And Related Diseases"             : 7,
    "Rashes"                                                            : 7,
    "Scabies Lyme Disease And Other Infestations And Bites"             : None,
    "Seborrheic Keratoses And Other Benign Tumors"                      : 4,
    "Systemic Disease"                                                  : 7,
    "Tinea Ringworm Candidiasis And Other Fungal Infections"            : None,
    "Urticaria Hives"                                                   : 7,
    "Vascular Tumors"                                                   : 5,
    "Vasculitis Photos"                                                 : 5,
    "Vi Chickenpox"                                                     : None,
    "Vi Shingles"                                                       : None,
    "Warts Molluscum And Other Viral Infections"                        : None,
}

# ══════════════════════════════════════════════════════════════════
# SECTION 2 — Helper
# ══════════════════════════════════════════════════════════════════

def add_row(rows, image_path, label, source, group_id):
    """Only add if image physically exists on disk."""
    if image_path and os.path.exists(image_path):
        rows.append({
            "image_path"   : str(image_path),
            "unified_label": label,
            "unified_name" : UNIFIED_CLASSES[label],
            "source"       : source,
            "group_id"     : str(group_id),
        })
        return True
    return False

# ══════════════════════════════════════════════════════════════════
# SECTION 3 — Load HAM10000
# ══════════════════════════════════════════════════════════════════

print(f"\n{'═'*60}")
print(f"  MAAI-SkinDx | Agent 3 — Data Preparation")
print(f"{'═'*60}\n")

rows = []

print("── Dataset 1: HAM10000 ──")
ham_loaded = 0
ham_skip   = 0

if not os.path.exists(HAM_META):
    print(f"  ERROR: {HAM_META} not found")
else:
    meta = pd.read_csv(HAM_META, sep=",")
    for _, row in meta.iterrows():
        label = HAM_MAP.get(row["dx"])
        if label is None:
            ham_skip += 1
            continue
        img = None
        for folder in [HAM_PART1, HAM_PART2]:
            p = os.path.join(folder, f"{row['image_id']}.jpg")
            if os.path.exists(p):
                img = p
                break
        if add_row(rows, img, label, "HAM10000", row["lesion_id"]):
            ham_loaded += 1
        else:
            ham_skip += 1

print(f"  Loaded  : {ham_loaded:,}")
print(f"  Skipped : {ham_skip:,}")

# ══════════════════════════════════════════════════════════════════
# SECTION 4 — Load Fitzpatrick-17k
# ══════════════════════════════════════════════════════════════════

print("\n── Dataset 2: Fitzpatrick-17k ──")
fitz_loaded   = 0
fitz_skip     = 0
fitz_unmapped = set()

if not os.path.exists(FITZ_CSV):
    print(f"  ERROR: {FITZ_CSV} not found")
else:
    fitz_df = pd.read_csv(FITZ_CSV)

    # Build md5 → image_path lookup from all 3 group folders
    md5_to_path = {}
    for folder in [FITZ_G1, FITZ_G2, FITZ_G3]:
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                md5 = os.path.splitext(fname)[0]
                md5_to_path[md5] = os.path.join(folder, fname)

    print(f"  Images on disk : {len(md5_to_path):,}")

    for _, row in fitz_df.iterrows():
        md5       = str(row["md5hash"]).strip()
        label_str = str(row["label"]).lower().strip()
        label     = FITZ_MAP.get(label_str)

        if label is None:
            fitz_unmapped.add(label_str)
            fitz_skip += 1
            continue

        img = md5_to_path.get(md5)
        if add_row(rows, img, label, "Fitzpatrick-17k", md5):
            fitz_loaded += 1
        else:
            fitz_skip += 1

    print(f"  Loaded  : {fitz_loaded:,}")
    print(f"  Skipped : {fitz_skip:,}")
    if fitz_unmapped:
        print(f"  Unmapped labels ({len(fitz_unmapped)} unique):")
        for l in sorted(list(fitz_unmapped))[:8]:
            print(f"    '{l}'")
        if len(fitz_unmapped) > 8:
            print(f"    ... and {len(fitz_unmapped)-8} more")

# ══════════════════════════════════════════════════════════════════
# SECTION 5 — Load PAD-UFES-20
# ══════════════════════════════════════════════════════════════════

print("\n── Dataset 3: PAD-UFES-20 ──")
pad_loaded = 0
pad_skip   = 0

if not os.path.exists(PAD_CSV):
    print(f"  ERROR: {PAD_CSV} not found")
else:
    pad_df = pd.read_csv(PAD_CSV)
    print(f"  CSV rows : {len(pad_df):,}")
    print(f"  Labels   : {pad_df['diagnostic'].unique().tolist()}")

    # Build filename → path lookup from all 3 image folders
    img_lookup = {}
    for folder in [PAD_PART1, PAD_PART2, PAD_PART3]:
        if not os.path.exists(folder):
            continue
        for fname in os.listdir(folder):
            img_lookup[fname] = os.path.join(folder, fname)

    print(f"  Images on disk : {len(img_lookup):,}")

    for _, row in pad_df.iterrows():
        label_str = str(row["diagnostic"]).strip()
        label     = PAD_MAP.get(label_str)

        if label is None:
            pad_skip += 1
            continue

        img_name = str(row["img_id"]).strip()
        img_path = img_lookup.get(img_name)

        if add_row(rows, img_path, label, "PAD-UFES-20", img_name):
            pad_loaded += 1
        else:
            pad_skip += 1

    print(f"  Loaded  : {pad_loaded:,}")
    print(f"  Skipped : {pad_skip:,}")

# ══════════════════════════════════════════════════════════════════
# SECTION 6 — Load Massive Balanced
# ══════════════════════════════════════════════════════════════════

print("\n── Dataset 4: Massive Balanced ──")
massive_loaded  = 0
massive_skip    = 0
massive_mapped  = 0
massive_skipped_folders = []

if not os.path.exists(MASSIVE_ROOT):
    print(f"  ERROR: {MASSIVE_ROOT} not found")
else:
    for folder_name in os.listdir(MASSIVE_ROOT):
        folder_path = os.path.join(MASSIVE_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            continue

        label = MASSIVE_MAP.get(folder_name)

        if label is None:
            massive_skipped_folders.append(folder_name)
            massive_skip += len(os.listdir(folder_path))
            continue

        massive_mapped += 1
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img_path = os.path.join(folder_path, fname)
            if add_row(rows, img_path, label,
                       "MassiveBalanced", fname):
                massive_loaded += 1
            else:
                massive_skip += 1

    print(f"  Folders mapped  : {massive_mapped}")
    print(f"  Folders skipped : {len(massive_skipped_folders)}")
    print(f"  Images loaded   : {massive_loaded:,}")
    print(f"  Images skipped  : {massive_skip:,}")
    print(f"  Skipped folders :")
    for f in massive_skipped_folders:
        print(f"    {f}")

# ══════════════════════════════════════════════════════════════════
# SECTION 7 — Prepare DDI-31 test set separately
# (TEST ONLY — never mixed into training data)
# ══════════════════════════════════════════════════════════════════

print("\n── Dataset 5: DDI-31 (TEST ONLY) ──")
ddi_rows   = []
ddi_loaded = 0
ddi_skip   = 0

if not os.path.exists(DDI_CSV):
    print(f"  ERROR: {DDI_CSV} not found")
else:
    ddi_df = pd.read_csv(DDI_CSV)
    print(f"  CSV rows : {len(ddi_df):,}")
    print(f"  Columns  : {list(ddi_df.columns)}")

    for _, row in ddi_df.iterrows():
        img_name = str(row["DDI_file"]).strip()
        img_path = os.path.join(DDI_IMGS, img_name)
        disease  = str(row["disease"]).strip()
        skin_tone= str(row["skin_tone"]).strip()
        malignant= str(row["malignant"]).strip()

        if os.path.exists(img_path):
            ddi_rows.append({
                "image_path": img_path,
                "disease"   : disease,
                "malignant" : malignant,
                "skin_tone" : skin_tone,
                "DDI_file"  : img_name,
            })
            ddi_loaded += 1
        else:
            ddi_skip += 1

    ddi_test_df = pd.DataFrame(ddi_rows)
    ddi_test_df.to_csv(OUTPUT_DDI_CSV, index=False)
    print(f"  Loaded  : {ddi_loaded:,}")
    print(f"  Skipped : {ddi_skip:,}")
    print(f"  Saved   : {OUTPUT_DDI_CSV}")

# ══════════════════════════════════════════════════════════════════
# SECTION 8 — Combine, validate, save
# ══════════════════════════════════════════════════════════════════

print(f"\n{'─'*60}")

if len(rows) == 0:
    print("  ERROR: No images loaded. Check dataset paths.")
    sys.exit(1)

unified_df = pd.DataFrame(rows)
total      = len(unified_df)

print(f"\n  COMBINED TRAINING DATASET SUMMARY")
print(f"  {'─'*40}")
print(f"  Total images : {total:,}\n")

print(f"  Per dataset:")
for src, cnt in unified_df["source"].value_counts().items():
    pct = cnt / total * 100
    print(f"    {src:<22} : {cnt:7,}  ({pct:4.1f}%)")

print(f"\n  Per unified class:")
for cls_idx in sorted(unified_df["unified_label"].unique()):
    cnt  = (unified_df["unified_label"] == cls_idx).sum()
    pct  = cnt / total * 100
    name = UNIFIED_CLASSES[cls_idx]
    bar  = "█" * int(pct / 2)
    print(f"    {cls_idx}: {name:<45} {cnt:6,} ({pct:4.1f}%)  {bar}")

# Check imbalance
counts    = unified_df["unified_label"].value_counts()
max_cnt   = counts.max()
min_cnt   = counts.min()
ratio     = max_cnt / max(min_cnt, 1)
print(f"\n  Imbalance ratio : {ratio:.1f}:1")
if ratio > 20:
    print(f"  WARNING: High imbalance detected.")
    print(f"    WeightedRandomSampler + Focal Loss gamma=3.0 will handle this.")

# Save unified training CSV
unified_df.to_csv(OUTPUT_CSV, index=False)
print(f"\n  Training CSV saved : {OUTPUT_CSV}")

# Save label map
pd.DataFrame([
    {"class_index": k, "class_name": v}
    for k, v in UNIFIED_CLASSES.items()
    if k in unified_df["unified_label"].unique()
]).to_csv(OUTPUT_LABEL_MAP, index=False)
print(f"  Label map saved    : {OUTPUT_LABEL_MAP}")

print(f"\n{'═'*60}")
print(f"  Data preparation complete.")
print(f"\n  Files created:")
print(f"    data/processed/agent3_unified_dataset.csv")
print(f"    data/processed/agent3_ddi31_test.csv")
print(f"    data/processed/agent3_label_mapping.csv")
print(f"\n  Next step:")
print(f"    python scripts/train_agent3_final.py")
print(f"{'═'*60}\n")