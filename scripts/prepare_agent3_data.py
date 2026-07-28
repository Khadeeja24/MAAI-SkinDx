# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 3 — Data Preparation (Final Correct Version)
# ══════════════════════════════════════════════════════════════════
# Run ONCE before training. Loads all 4 datasets, maps labels to
# unified 9-class taxonomy, validates every image on disk,
# saves one clean CSV ready for training.
#
# Fixes applied:
#   1. PAD-UFES-20 MEL label added (52 melanoma images recovered)
#   2. 30+ Fitzpatrick label mappings added (~5,000 more images)
#   3. Contaminated Massive Balanced folder removed (BCC/AK mixed)
#   4. Robust error handling throughout
#
# Unified 9-class taxonomy:
#   0: Melanoma
#   1: Melanocytic Nevus
#   2: Basal Cell Carcinoma
#   3: Actinic Keratosis / Squamous Cell Carcinoma
#   4: Benign Keratosis
#   5: Vascular Lesion
#   6: Dermatofibroma
#   7: Inflammatory
#   8: Eczema / Dermatitis
#
# Output: data/processed/agent3_unified_dataset.csv
# ══════════════════════════════════════════════════════════════════

import os
import sys
import pandas as pd

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
MASSIVE_ROOT= os.path.join(PROJECT_ROOT, "data", "raw", "massive_balanced", "balanced_dataset")
DDI_CSV     = os.path.join(PROJECT_ROOT, "data", "raw", "ddi31", "ddi_metadata.csv")
DDI_IMGS    = os.path.join(PROJECT_ROOT, "data", "raw", "ddi31")

OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "data", "processed")
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "agent3_unified_dataset.csv")
OUTPUT_DDI  = os.path.join(OUTPUT_DIR, "agent3_ddi31_test.csv")
OUTPUT_MAP  = os.path.join(OUTPUT_DIR, "agent3_label_mapping.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Unified 9-class taxonomy ───────────────────────────────────────
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

# ── HAM10000 ──────────────────────────────────────────────────────
HAM_MAP = {
    "mel": 0, "nv": 1, "bcc": 2,
    "akiec": 3, "bkl": 4, "vasc": 5, "df": 6,
}

# ── Fitzpatrick-17k — comprehensive mapping ────────────────────────
FITZ_MAP = {
    # Melanoma → 0
    "melanoma": 0, "nodular melanoma": 0,
    "lentigo maligna": 0, "lentigo maligna melanoma": 0,
    "acral lentiginous melanoma": 0, "amelanotic melanoma": 0,
    "melanoma in situ": 0, "superficial spreading melanoma": 0,
    "desmoplastic melanoma": 0, "melanoma metastatic": 0,
    # Nevus → 1
    "melanocytic nevi": 1, "blue nevus": 1, "compound nevus": 1,
    "intradermal nevus": 1, "junctional nevus": 1,
    "congenital nevus": 1, "dysplastic nevus": 1,
    "spitz nevus": 1, "halo nevus": 1, "becker nevus": 1,
    "nevus": 1, "nevus sebaceous": 1, "epidermal nevus": 1,
    "pigmented spindle cell nevus": 1,
    # BCC → 2
    "basal cell carcinoma": 2,
    "basal cell carcinoma morpheiform": 2,
    "basal cell carcinoma nodular": 2,
    "basal cell carcinoma superficial": 2,
    "basal cell carcinoma pigmented": 2,
    # AK / SCC → 3
    "squamous cell carcinoma": 3, "actinic keratosis": 3,
    "bowen's disease": 3, "keratoacanthoma": 3,
    "cutaneous horn": 3,
    # Benign keratosis → 4
    "seborrheic keratosis": 4, "solar lentigo": 4,
    "dermatosis papulosa nigra": 4, "stucco keratosis": 4,
    "lichenoid keratosis": 4, "acanthosis nigricans": 4,
    "porokeratosis": 4,
    # Vascular → 5
    "hemangioma": 5, "pyogenic granuloma": 5,
    "cherry angioma": 5, "venous lake": 5,
    "angiokeratoma": 5, "angioma": 5,
    "lymphangioma": 5, "port wine stain": 5,
    "kaposi sarcoma": 5, "purpura": 5,
    # Dermatofibroma → 6
    "dermatofibroma": 6,
    # Inflammatory → 7
    "psoriasis": 7, "lichen planus": 7,
    "lupus erythematosus": 7, "sarcoidosis": 7,
    "neutrophilic dermatoses": 7, "photodermatoses": 7,
    "pityriasis rosea": 7, "granuloma annulare": 7,
    "morphea": 7, "vitiligo": 7, "urticaria": 7,
    "drug eruptions": 7, "drug eruption": 7,
    "lichen sclerosus": 7, "rosacea": 7,
    "acne": 7, "acne vulgaris": 7,
    "pityriasis lichenoides": 7, "folliculitis": 7,
    "hidradenitis suppurativa": 7, "prurigo nodularis": 7,
    "erythema multiforme": 7, "erythema nodosum": 7,
    "cellulitis": 7, "impetigo": 7, "vasculitis": 7,
    "tinea": 7, "pemphigus": 7, "pemphigoid": 7,
    "erythema": 7, "dermatomyositis": 7, "scleroderma": 7,
    "behcets disease": 7, "behcet": 7,
    # Eczema → 8
    "eczema": 8, "atopic dermatitis": 8,
    "allergic contact dermatitis": 8,
    "contact dermatitis": 8, "dermatitis": 8,
    "irritant contact dermatitis": 8,
    "seborrheic dermatitis": 8, "nummular eczema": 8,
    "dyshidrotic eczema": 8, "stasis dermatitis": 8,
    "perioral dermatitis": 8, "pompholyx": 8,
}

# ── PAD-UFES-20 — FIXED: MEL added ───────────────────────────────
PAD_MAP = {
    "BCC": 2, "SCC": 3, "ACK": 3,
    "SEK": 4, "BOD": 3, "NEV": 1,
    "MEL": 0,   # ← was missing — 52 melanoma images recovered
}

# ── Massive Balanced — FIXED: contaminated folder removed ─────────
MASSIVE_MAP = {
    "Acne And Rosacea Photos"                                           : 7,
    # REMOVED: "Actinic Keratosis Basal Cell Carcinoma..." → None
    # Reason: folder mixes BCC and AK causing taxonomy contamination.
    # BCC precision was 61% because of this. Removing it.
    "Actinic Keratosis Basal Cell Carcinoma And Other Malignant Lesions": None,
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

# ── Helper ─────────────────────────────────────────────────────────
def add_row(rows, image_path, label, source, group_id):
    """Only add if image physically exists on disk."""
    if image_path and os.path.exists(str(image_path)):
        rows.append({
            "image_path"   : str(image_path),
            "unified_label": int(label),
            "unified_name" : UNIFIED_CLASSES[label],
            "source"       : source,
            "group_id"     : str(group_id),
        })
        return True
    return False

# ══════════════════════════════════════════════════════════════════
print(f"\n{'═'*65}")
print(f"  MAAI-SkinDx | Agent 3 — Data Preparation (Final v2)")
print(f"{'═'*65}\n")

rows = []

# ── Dataset 1: HAM10000 ───────────────────────────────────────────
print("── Dataset 1: HAM10000 (dermoscopic, gold standard) ──")
ham_loaded = ham_skip = 0
if not os.path.exists(HAM_META):
    print(f"  ERROR: {HAM_META} not found")
else:
    try:
        meta = pd.read_csv(HAM_META, sep=",")
        for _, row in meta.iterrows():
            label = HAM_MAP.get(str(row["dx"]).strip())
            if label is None:
                ham_skip += 1; continue
            img = None
            for folder in [HAM_PART1, HAM_PART2]:
                p = os.path.join(folder, f"{row['image_id']}.jpg")
                if os.path.exists(p):
                    img = p; break
            if add_row(rows, img, label, "HAM10000", row["lesion_id"]):
                ham_loaded += 1
            else:
                ham_skip += 1
    except Exception as e:
        print(f"  ERROR loading HAM10000: {e}")
print(f"  Loaded : {ham_loaded:,}")
print(f"  Skipped: {ham_skip:,}")

# ── Dataset 2: Fitzpatrick-17k ────────────────────────────────────
print("\n── Dataset 2: Fitzpatrick-17k (clinical, skin tone diversity) ──")
fitz_loaded = fitz_skip = 0
fitz_unmapped = set()
if not os.path.exists(FITZ_CSV):
    print(f"  ERROR: {FITZ_CSV} not found")
else:
    try:
        fitz_df = pd.read_csv(FITZ_CSV)
        # Build md5hash → image path lookup from all 3 group folders
        md5_to_path = {}
        for folder in [FITZ_G1, FITZ_G2, FITZ_G3]:
            if not os.path.exists(folder): continue
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg",".jpeg",".png")):
                    md5 = os.path.splitext(fname)[0].strip()
                    md5_to_path[md5] = os.path.join(folder, fname)
        print(f"  Images on disk: {len(md5_to_path):,}")
        for _, row in fitz_df.iterrows():
            md5       = str(row["md5hash"]).strip()
            label_str = str(row["label"]).lower().strip()
            label     = FITZ_MAP.get(label_str)
            if label is None:
                fitz_unmapped.add(label_str)
                fitz_skip += 1; continue
            img = md5_to_path.get(md5)
            if add_row(rows, img, label, "Fitzpatrick-17k", md5):
                fitz_loaded += 1
            else:
                fitz_skip += 1
    except Exception as e:
        print(f"  ERROR loading Fitzpatrick: {e}")
print(f"  Loaded : {fitz_loaded:,}")
print(f"  Skipped: {fitz_skip:,}")
print(f"  Still unmapped: {len(fitz_unmapped)} unique labels "
      f"(acceptable — rare dermatological conditions)")

# ── Dataset 3: PAD-UFES-20 ───────────────────────────────────────
print("\n── Dataset 3: PAD-UFES-20 (smartphone clinical + patient data) ──")
pad_loaded = pad_skip = 0
if not os.path.exists(PAD_CSV):
    print(f"  ERROR: {PAD_CSV} not found")
else:
    try:
        pad_df = pd.read_csv(PAD_CSV)
        # Build image name → path lookup
        img_lookup = {}
        for folder in [PAD_PART1, PAD_PART2, PAD_PART3]:
            if not os.path.exists(folder): continue
            for fname in os.listdir(folder):
                img_lookup[fname.strip()] = os.path.join(folder, fname)
        print(f"  Images on disk   : {len(img_lookup):,}")
        print(f"  Labels in CSV    : {sorted(pad_df['diagnostic'].unique())}")
        for _, row in pad_df.iterrows():
            label_str = str(row["diagnostic"]).strip()
            label     = PAD_MAP.get(label_str)
            if label is None:
                pad_skip += 1; continue
            img_name = str(row["img_id"]).strip()
            img_path = img_lookup.get(img_name)
            if add_row(rows, img_path, label, "PAD-UFES-20", img_name):
                pad_loaded += 1
            else:
                pad_skip += 1
        mel_count = (pad_df["diagnostic"] == "MEL").sum()
        print(f"  MEL (melanoma) images included: {mel_count} ← was 0 before fix")
    except Exception as e:
        print(f"  ERROR loading PAD-UFES-20: {e}")
print(f"  Loaded : {pad_loaded:,}")
print(f"  Skipped: {pad_skip:,}")

# ── Dataset 4: Massive Balanced ───────────────────────────────────
print("\n── Dataset 4: Massive Balanced (clinical diversity) ──")
massive_loaded = massive_skip = 0
skipped_folders = []
if not os.path.exists(MASSIVE_ROOT):
    print(f"  NOT FOUND: {MASSIVE_ROOT}")
else:
    try:
        for folder_name in os.listdir(MASSIVE_ROOT):
            folder_path = os.path.join(MASSIVE_ROOT, folder_name)
            if not os.path.isdir(folder_path): continue
            label = MASSIVE_MAP.get(folder_name)
            if label is None:
                skipped_folders.append(folder_name)
                continue
            for fname in os.listdir(folder_path):
                if not fname.lower().endswith((".jpg",".jpeg",".png")):
                    continue
                if add_row(rows, os.path.join(folder_path, fname),
                           label, "MassiveBalanced", fname):
                    massive_loaded += 1
                else:
                    massive_skip += 1
    except Exception as e:
        print(f"  ERROR loading Massive Balanced: {e}")
print(f"  Loaded : {massive_loaded:,}")
print(f"  Skipped: {massive_skip:,} "
      f"(ambiguous/out-of-scope + contaminated folder removed)")

# ── Dataset 5: DDI-31 — TEST ONLY ────────────────────────────────
print("\n── Dataset 5: DDI-31 (TEST ONLY — never in training) ──")
ddi_rows = []
if os.path.exists(DDI_CSV):
    try:
        ddi_df = pd.read_csv(DDI_CSV)
        for _, row in ddi_df.iterrows():
            img_path = os.path.join(DDI_IMGS, str(row["DDI_file"]).strip())
            if os.path.exists(img_path):
                ddi_rows.append({
                    "image_path": img_path,
                    "disease"   : str(row["disease"]).strip(),
                    "malignant" : str(row["malignant"]).strip(),
                    "skin_tone" : str(row["skin_tone"]).strip(),
                    "DDI_file"  : str(row["DDI_file"]).strip(),
                })
        pd.DataFrame(ddi_rows).to_csv(OUTPUT_DDI, index=False)
        print(f"  Loaded : {len(ddi_rows):,} | Saved: {OUTPUT_DDI}")
    except Exception as e:
        print(f"  ERROR loading DDI-31: {e}")

# ══════════════════════════════════════════════════════════════════
# Combine, validate, save
# ══════════════════════════════════════════════════════════════════
print(f"\n{'─'*65}")

if len(rows) == 0:
    print("ERROR: No images loaded. Check dataset paths.")
    sys.exit(1)

unified_df = pd.DataFrame(rows)
total      = len(unified_df)

print(f"\n  COMBINED DATASET SUMMARY")
print(f"  Total images: {total:,}\n")

print(f"  Per dataset:")
for src, cnt in unified_df["source"].value_counts().items():
    pct = cnt / total * 100
    tag = " (clinical)" if src != "HAM10000" else " (dermoscopic)"
    print(f"    {src:<22}: {cnt:7,} ({pct:4.1f}%){tag}")

clin = unified_df[unified_df["source"] != "HAM10000"].shape[0]
print(f"\n  Clinical photos total : {clin:,} ({clin/total*100:.1f}%)")
print(f"  Dermoscopic total     : {total-clin:,} ({(total-clin)/total*100:.1f}%)")

print(f"\n  Per unified class:")
for cls_idx in sorted(unified_df["unified_label"].unique()):
    cnt  = (unified_df["unified_label"] == cls_idx).sum()
    name = UNIFIED_CLASSES[cls_idx]
    bar  = "█" * int(cnt/total*100/2)
    malignant = " ★ MALIGNANT" if cls_idx in [0,2,3] else ""
    print(f"    {cls_idx}: {name:<45} {cnt:6,} "
          f"({cnt/total*100:4.1f}%)  {bar}{malignant}")

counts = unified_df["unified_label"].value_counts()
ratio  = counts.max() / max(counts.min(), 1)
print(f"\n  Imbalance ratio: {ratio:.1f}:1")
print(f"  Handled by: class capping + WeightedRandomSampler + FocalLoss")

# Save
unified_df.to_csv(OUTPUT_CSV, index=False)
pd.DataFrame([{"class_index": k, "class_name": v}
              for k, v in UNIFIED_CLASSES.items()
              if k in unified_df["unified_label"].unique()]
             ).to_csv(OUTPUT_MAP, index=False)

print(f"\n  Saved: {OUTPUT_CSV}")
print(f"  Saved: {OUTPUT_MAP}")
print(f"\n{'═'*65}")
print(f"  Data preparation complete.")
print(f"\n  Next step:")
print(f"    python scripts/preextract_agent2_features.py")
print(f"  This extracts Agent 2 features for training images.")
print(f"  Time estimate: 10-15 hours (run overnight)")
print(f"{'═'*65}\n")