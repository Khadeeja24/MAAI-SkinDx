# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 1 — Image Quality Agent
# ══════════════════════════════════════════════════════════════════
# Entry point for all image quality validation in the pipeline.
# Every image must pass through this agent before reaching
# Agent 2 (Visual Feature Extraction).
#
# Checks performed (in order):
#   1. Blur Detection       — Laplacian variance
#   2. Quality Assessment   — BRISQUE score
#   3. Skin Detection       — Circular mask + YCbCr
#
# Returns a structured result dict with status, reason, and scores.
# The Orchestrator communicates with Agent 1 through this file only.
# ══════════════════════════════════════════════════════════════════

from .checks.blur_check    import check_blur
from .checks.quality_check import check_quality
from .checks.skin_check    import check_skin
from .utils                import load_image, resize_image


class ImageQualityAgent:

    def __init__(self):
        self.name = "Agent 1 — Image Quality Agent"
        print(f"[{self.name}] Initialised")

    # ─── Main Entry Point ──────────────────────────────────────────

    def run(self, image_path: str) -> dict:
        """
        Execute all quality checks on the provided image.

        Args:
            image_path (str): Path to the input image file.

        Returns:
            dict:
                status  (str)  : 'PASS' or 'FAIL'
                reason  (str)  : Human-readable outcome message.
                scores  (dict) : Numeric scores from each check.
        """
        print(f"\n{'═' * 60}")
        print(f"  {self.name}")
        print(f"  Image : {image_path}")
        print(f"{'═' * 60}")

        # ── Load ───────────────────────────────────────────────────
        image = load_image(image_path)
        if image is None:
            return self._result(
                "FAIL",
                "Image could not be loaded. File may be corrupted or the format is not supported.",
                {}
            )

        image = resize_image(image, max_size=512)

        # ── Check 1: Blur ──────────────────────────────────────────
        is_blurry, blur_score = check_blur(image)
        if is_blurry:
            return self._result(
                "FAIL",
                f"Image is too blurry (variance: {blur_score}). Please retake with a steady hand.",
                {"blur_variance": blur_score}
            )

        # ── Check 2: Quality ───────────────────────────────────────
        is_poor_quality, brisque_score = check_quality(image)
        if is_poor_quality:
            return self._result(
                "FAIL",
                f"Image quality is too low (BRISQUE: {brisque_score}). Please use better lighting.",
                {
                    "blur_variance" : blur_score,
                    "brisque_score" : brisque_score,
                }
            )

        # ── Check 3: Skin ──────────────────────────────────────────
        has_skin, method, skin_details = check_skin(image)
        if not has_skin:
            return self._result(
                "FAIL",
                "No skin detected. Please upload a clear photograph of the affected skin area.",
                {
                    "blur_variance" : blur_score,
                    "brisque_score" : brisque_score,
                    **skin_details,
                }
            )

        # ── All Checks Passed ──────────────────────────────────────
        print(f"\n  ✓ PASS — Image approved for diagnostic pipeline")
        print(f"{'═' * 60}\n")

        return self._result(
            "PASS",
            "Image passed all quality checks.",
            {
                "blur_variance"          : blur_score,
                "brisque_score"          : brisque_score,
                "skin_detection_method"  : method,
                **skin_details,
            }
        )

    # ─── Internal Helper ───────────────────────────────────────────

    def _result(self, status: str, reason: str, scores: dict) -> dict:
        """Build and return a structured result dictionary."""
        print(f"\n  Status : {status}")
        print(f"  Reason : {reason}")
        return {
            "status" : status,
            "reason" : reason,
            "scores" : scores,
        }