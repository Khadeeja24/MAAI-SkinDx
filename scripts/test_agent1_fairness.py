# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 1 — Skin Tone Fairness Test
# ══════════════════════════════════════════════════════════════════
# Tests Agent 1 across three Fitzpatrick skin tone groups:
#   group1_light  → Fitzpatrick Type 1 and 2
#   group2_medium → Fitzpatrick Type 3 and 4
#   group3_dark   → Fitzpatrick Type 5 and 6
#
# All images are real clinical skin disease photos.
# All should PASS Agent 1 skin detection.
# Per-group accuracy reveals any skin tone bias.
# ══════════════════════════════════════════════════════════════════

import os
import sys
import random
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from agents.agent1_image_quality import ImageQualityAgent

# ─── Paths ─────────────────────────────────────────────────────────
FITZ_BASE = os.path.join(
    PROJECT_ROOT, "data", "raw",
    "fitzpatrick17k", "fitzpatrick_3groups"
)

# ─── Skin tone groups ──────────────────────────────────────────────
GROUPS = {
    "group1_light  (Fitzpatrick Type 1-2)" : "group1_light",
    "group2_medium (Fitzpatrick Type 3-4)" : "group2_medium",
    "group3_dark   (Fitzpatrick Type 5-6)" : "group3_dark",
}

# Test 50 images per group — enough for meaningful stats
IMAGES_PER_GROUP = 50
random.seed(42)

print(f"\n{'═' * 60}")
print(f"  MAAI-SkinDx | Agent 1 — Skin Tone Fairness Test")
print(f"{'═' * 60}")
print(f"  Testing {IMAGES_PER_GROUP} images per skin tone group")
print(f"  All images are real clinical skin photos — all should PASS")
print(f"  Any group below 90% PASS rate indicates skin tone bias")
print(f"{'═' * 60}\n")

# ─── Initialise Agent 1 ────────────────────────────────────────────
agent = ImageQualityAgent()

# ─── Run test per group ────────────────────────────────────────────
group_results = {}

for group_label, group_folder in GROUPS.items():
    folder_path = os.path.join(FITZ_BASE, group_folder)

    # Collect all images in this group
    all_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        all_images.extend(list(Path(folder_path).rglob(ext)))

    # Deduplicate by lowercase name
    seen      = set()
    unique    = []
    for img in all_images:
        key = img.name.lower()
        if key not in seen:
            seen.add(key)
            unique.append(img)

    # Pick random sample
    sample = random.sample(unique, min(IMAGES_PER_GROUP, len(unique)))

    print(f"── {group_label} ──")
    print(f"   Total images in group : {len(unique)}")
    print(f"   Testing               : {len(sample)} images")

    passed = 0
    failed = 0
    failed_images = []

    for img_path in sample:
        result = agent.run(str(img_path))
        if result['status'] == 'PASS':
            passed += 1
        else:
            failed += 1
            failed_images.append({
                'image'      : img_path.name,
                'reason'     : result['reason'],
                'confidence' : result.get('scores', {}).get('skin_confidence', 'N/A')
            })

    pass_rate = passed / len(sample) * 100

    group_results[group_label] = {
        'total'     : len(sample),
        'passed'    : passed,
        'failed'    : failed,
        'pass_rate' : pass_rate,
    }

    print(f"   Passed : {passed}/{len(sample)}")
    print(f"   Failed : {failed}/{len(sample)}")
    print(f"   Pass rate : {pass_rate:.1f}%")

    if failed_images:
        print(f"   Failed images:")
        for f in failed_images[:3]:
            print(f"     {f['image']} — {f['reason']} "
                  f"(confidence: {f['confidence']})")

    print()

# ─── Final fairness summary ────────────────────────────────────────
total_passed = sum(r['passed'] for r in group_results.values())
total_images = sum(r['total']  for r in group_results.values())
overall_acc  = total_passed / total_images * 100

pass_rates   = [r['pass_rate'] for r in group_results.values()]
max_gap      = max(pass_rates) - min(pass_rates)

print(f"{'═' * 60}")
print(f"  FAIRNESS TEST SUMMARY")
print(f"{'═' * 60}")
for label, result in group_results.items():
    bar_fill = int(result['pass_rate'] / 2)
    bar      = '█' * bar_fill + '░' * (50 - bar_fill)
    print(f"  {label}")
    print(f"    [{bar}] {result['pass_rate']:.1f}%")
    print()

print(f"  Overall accuracy    : {total_passed}/{total_images} = {overall_acc:.1f}%")
print(f"  Max gap between groups : {max_gap:.1f}%")
print()

if max_gap <= 5:
    fairness = "FAIR — gap between skin tone groups is within 5%"
elif max_gap <= 10:
    fairness = "ACCEPTABLE — small gap, monitor on larger dataset"
else:
    fairness = "BIAS DETECTED — significant gap between skin tone groups"

print(f"  Fairness verdict    : {fairness}")
print(f"{'═' * 60}\n")