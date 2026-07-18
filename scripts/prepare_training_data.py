# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Skin Detector — Training Data Preparation
# ══════════════════════════════════════════════════════════════════

import os
import shutil
import random
import pandas as pd
import requests
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW       = os.path.join(PROJECT_ROOT, "data", "raw")

HAM_PART1      = os.path.join(DATA_RAW, "ham10000", "HAM10000_images_part_1")
HAM_PART2      = os.path.join(DATA_RAW, "ham10000", "HAM10000_images_part_2")
HAM_META       = os.path.join(DATA_RAW, "ham10000", "HAM10000_metadata")

FITZ_CSV       = os.path.join(DATA_RAW, "fitzpatrick17k", "fitzpatrick17k.csv")

DTD_IMAGES     = os.path.join(DATA_RAW, "skin_detector_training", "not_skin", "dtd", "dtd", "images")
MY_PHOTOS      = os.path.join(DATA_RAW, "skin_detector_training", "not_skin", "my_photos")

OUT_DERM       = os.path.join(DATA_RAW, "skin_detector_training", "skin", "dermoscopic")
OUT_CLIN       = os.path.join(DATA_RAW, "skin_detector_training", "skin", "clinical")
OUT_NOT_SKIN   = os.path.join(DATA_RAW, "skin_detector_training", "not_skin", "selected")

# Create output folders
os.makedirs(OUT_DERM,     exist_ok=True)
os.makedirs(OUT_CLIN,     exist_ok=True)
os.makedirs(OUT_NOT_SKIN, exist_ok=True)

random.seed(42)

# ══════════════════════════════════════════════════════════════════
# STEP 1 — Pick 500 dermoscopic images from HAM10000
# ══════════════════════════════════════════════════════════════════
print("\n── Step 1: Picking 500 dermoscopic images from HAM10000 ──")

# Skip if already done
existing_derm = len(list(Path(OUT_DERM).glob("*.jpg")))
if existing_derm >= 500:
    print(f"  Already have {existing_derm} dermoscopic images — skipping")
else:
    meta = pd.read_csv(HAM_META, sep=',')
    print(f"  Total HAM10000 images: {len(meta)}")
    print(f"  Disease categories: {meta['dx'].unique()}")

    categories   = meta['dx'].unique()
    per_category = 500 // len(categories)
    selected_ids = []

    for cat in categories:
        cat_ids = meta[meta['dx'] == cat]['image_id'].tolist()
        picked  = random.sample(cat_ids, min(per_category, len(cat_ids)))
        selected_ids.extend(picked)

    remaining = meta[~meta['image_id'].isin(selected_ids)]['image_id'].tolist()
    while len(selected_ids) < 500:
        selected_ids.append(random.choice(remaining))

    selected_ids = selected_ids[:500]
    print(f"  Selected {len(selected_ids)} images")

    copied = 0
    for img_id in selected_ids:
        for folder in [HAM_PART1, HAM_PART2]:
            src = os.path.join(folder, f"{img_id}.jpg")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(OUT_DERM, f"{img_id}.jpg"))
                copied += 1
                break

    print(f"  Copied {copied} dermoscopic images to: {OUT_DERM}")


# ══════════════════════════════════════════════════════════════════
# STEP 2 — Pick 500 clinical images from Fitzpatrick 3 groups
# ══════════════════════════════════════════════════════════════════
print("\n── Step 2: Picking 500 clinical images from Fitzpatrick ──")

# Skip if already done
existing_clin = len(list(Path(OUT_CLIN).glob("*.*")))
if existing_clin >= 500:
    print(f"  Already have {existing_clin} clinical images — skipping")
else:
    FITZ_BASE = os.path.join(DATA_RAW, "fitzpatrick17k", "fitzpatrick_3groups")

    groups = {
        "group1_light"  : 167,
        "group2_medium" : 167,
        "group3_dark"   : 166,
    }

    total_copied = 0

    for group_name, target_count in groups.items():
        group_path = os.path.join(FITZ_BASE, group_name)

        # Collect all images in this group
        all_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            all_images.extend(list(Path(group_path).glob(ext)))

        # Deduplicate
        seen   = set()
        unique = []
        for img in all_images:
            key = img.name.lower()
            if key not in seen:
                seen.add(key)
                unique.append(img)

        # Pick random sample
        selected = random.sample(unique, min(target_count, len(unique)))

        # Copy to clinical folder
        for img_path in selected:
            out_path = os.path.join(OUT_CLIN, f"{group_name}_{img_path.name}")
            if not os.path.exists(out_path):
                shutil.copy2(img_path, out_path)
                total_copied += 1

        print(f"  {group_name}: picked {len(selected)} images")

    print(f"  Total clinical images copied: {total_copied}")
    print(f"  Clinical images saved to: {OUT_CLIN}")

# ══════════════════════════════════════════════════════════════════
# STEP 3 — Pick 800 texture images from DTD
# ══════════════════════════════════════════════════════════════════
print("\n── Step 3: Picking 800 texture images from DTD ──")

existing_not_skin = len(list(Path(OUT_NOT_SKIN).glob("*.*")))
if existing_not_skin >= 900:
    print(f"  Already have {existing_not_skin} not-skin images — skipping Steps 3 and 4")
else:
    if os.path.exists(OUT_NOT_SKIN):
        shutil.rmtree(OUT_NOT_SKIN)
    os.makedirs(OUT_NOT_SKIN)

    all_dtd_images = []
    for category_folder in Path(DTD_IMAGES).iterdir():
        if category_folder.is_dir():
            images = list(category_folder.glob("*.jpg"))
            all_dtd_images.extend(images)

    print(f"  Total DTD images found: {len(all_dtd_images)}")

    selected_dtd = random.sample(all_dtd_images, min(800, len(all_dtd_images)))

    for i, img_path in enumerate(selected_dtd):
        out_path = os.path.join(OUT_NOT_SKIN, f"dtd_{i:04d}.jpg")
        shutil.copy2(img_path, out_path)

    print(f"  Copied {len(selected_dtd)} DTD images to: {OUT_NOT_SKIN}")

    # ══════════════════════════════════════════════════════════════════
    # STEP 4 — Copy phone photos
    # ══════════════════════════════════════════════════════════════════
    print("\n── Step 4: Copying phone photos ──")

    phone_images = []
    seen_names   = set()

    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        for img in Path(MY_PHOTOS).glob(ext):
            name_lower = img.name.lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                phone_images.append(img)

    print(f"  Found {len(phone_images)} phone photos")

    for i, img_path in enumerate(phone_images):
        out_path = os.path.join(OUT_NOT_SKIN, f"phone_{i:04d}{img_path.suffix}")
        shutil.copy2(img_path, out_path)

    print(f"  Copied {len(phone_images)} phone photos to: {OUT_NOT_SKIN}")

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
derm_count     = len(list(Path(OUT_DERM).glob("*.jpg")))
clin_count     = len(list(Path(OUT_CLIN).glob("*.*")))
not_skin_count = len(list(Path(OUT_NOT_SKIN).glob("*.*")))

print(f"""
══════════════════════════════════════════
  Data Preparation Complete
══════════════════════════════════════════
  Dermoscopic (skin)  : {derm_count} images
  Clinical (skin)     : {clin_count} images
  Not skin            : {not_skin_count} images
  Total               : {derm_count + clin_count + not_skin_count} images
══════════════════════════════════════════
""")