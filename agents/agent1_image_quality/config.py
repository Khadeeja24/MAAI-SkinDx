# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 1 — Image Quality Agent
# Configuration
# ══════════════════════════════════════════════════════════════════

# ─── Blur Detection ───────────────────────────────────────────────
# Laplacian variance threshold.
# Below this → image too blurry → rejected.
BLUR_THRESHOLD = 100

# ─── Image Quality Assessment ─────────────────────────────────────
# BRISQUE score threshold. Lower = better quality.
# Above this → image poor quality → rejected.
BRISQUE_THRESHOLD = 40