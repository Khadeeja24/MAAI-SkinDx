# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 2 — Visual Feature Extraction Agent
# ══════════════════════════════════════════════════════════════════
# Receives approved image from Agent 1 via Orchestrator.
# Runs three feature extraction streams:
#   Stream A — DINOv2 ViT-B/14  → 768-dim  dermoscopic embedding
#   Stream B — Derm Foundation   → 6144-dim clinical embedding
#   Stream C — ABCD Scores       → 4-dim    clinical feature vector
#
# Model selection — evaluated by linear probe + k-NN probe:
#   Dermoscopic: DINOv2 ViT-B/14 won (k-NN F1 45.46% on HAM10000)
#     PanDerm 26.29% | ViT-B/16 37.47% | ResNet-50 38.68%
#   Clinical: Derm Foundation won (F1 76.83% on Fitzpatrick-17k)
#     ResNet-50 57.41% | EfficientNet-B4 57.62%
#
# Normalization applied per stream before combining:
#   DINOv2        → L2 normalization (unit vector)
#   Derm Foundation → L2 normalization (unit vector)
#   ABCD Scores    → MinMax then L2 normalization
#
# Equal stream weighting applied after normalization:
#   Each stream multiplied by 1/3 so all three streams contribute
#   equally regardless of dimension count (768 vs 6144 vs 4).
#
# Combined output: 6916-dim feature vector for Agent 3.
# ══════════════════════════════════════════════════════════════════

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .streams.dinov2_stream          import DINOv2Stream
from .streams.derm_foundation_stream import DermFoundationStream
from .streams.abcd_scores            import compute_abcd_scores
from .utils                          import load_image, preprocess_for_model
from .config                         import COMBINED_FEATURE_DIM


# Equal weight for each stream — prevents Derm Foundation (6144 dims)
# from dominating DINOv2 (768 dims) and ABCD (4 dims)
STREAM_WEIGHT = 1.0 / 3.0


class FeatureExtractionAgent:

    def __init__(self):
        self.name = "Agent 2 — Visual Feature Extraction Agent"
        print(f"\n[{self.name}] Initialising ...")

        self.dinov2          = DINOv2Stream()
        self.derm_foundation = DermFoundationStream()

        # GPU warm-up — eliminates slow first inference
        # DINOv2 and Derm Foundation both compile CUDA kernels
        # on their first forward pass. Running a dummy image now
        # means the first real patient image runs at full speed.
        print(f"  [Agent 2] Warming up GPU...")
        try:
            import tempfile
            import contextlib
            import io
            from PIL import Image as _PILImage
            dummy_arr  = np.zeros((224, 224, 3), dtype=np.uint8)
            dummy_img  = _PILImage.fromarray(dummy_arr)
            tmp_path   = os.path.join(
                tempfile.gettempdir(), "maai_warmup_a2.jpg")
            dummy_img.save(tmp_path)
            with contextlib.redirect_stdout(io.StringIO()):
                self.run(tmp_path)
            os.remove(tmp_path)
            print(f"  [Agent 2] GPU ready")
        except Exception:
            print(f"  [Agent 2] Warm-up skipped")

        print(f"[{self.name}] Ready")

    # ─── Normalization Helpers ─────────────────────────────────────

    @staticmethod
    def _l2_normalize(v: np.ndarray) -> np.ndarray:
        """
        L2 normalization — scales vector to unit length.
        After this, the vector has magnitude exactly 1.0.
        Standard normalization for deep learning embeddings.
        """
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    @staticmethod
    def _minmax_normalize(v: np.ndarray) -> np.ndarray:
        """
        MinMax normalization — scales all values to [0, 1] range.
        Used for ABCD scores because asymmetry, border, colour, and
        diameter are geometric measurements on completely different
        numerical scales.
        """
        vmin = v.min()
        vmax = v.max()
        return (v - vmin) / (vmax - vmin) if vmax > vmin else v

    # ─── Main Entry Point ──────────────────────────────────────────

    def run(self, image_path: str) -> dict:
        """
        Extract all features from an approved image.
        Called by the Orchestrator only — never called directly.

        Args:
            image_path (str): Path to image that passed Agent 1.

        Returns:
            dict:
                status         (str)           : 'PASS' or 'FAIL'
                feature_vector (numpy.ndarray) : Combined [6916] vector
                feature_dim    (int)           : Total dimensions
                components     (dict)          : Per-stream details
                reason         (str)           : Description
        """
        print(f"\n{'═' * 60}")
        print(f"  {self.name}")
        print(f"  Image : {image_path}")
        print(f"{'═' * 60}")

        # ── Load Image ─────────────────────────────────────────────
        image = load_image(image_path)
        if image is None:
            return {
                "status"        : "FAIL",
                "reason"        : "Could not load image.",
                "feature_vector": None,
                "feature_dim"   : 0,
                "components"    : {}
            }

        # ── Stream A: DINOv2 ───────────────────────────────────────
        print(f"\n  ── Stream A: DINOv2 ViT-B/14 (768-dim) ──")
        dinov2_raw = self.dinov2.extract(image)

        # ── Stream B: Derm Foundation ──────────────────────────────
        print(f"\n  ── Stream B: Derm Foundation (6144-dim) ──")
        derm_raw = self.derm_foundation.extract(image)

        # ── Stream C: ABCD Scores ──────────────────────────────────
        print(f"\n  ── Stream C: ABCD Scores (4-dim) ──")
        abcd_raw = compute_abcd_scores(image)

        # ── Step 1: Normalize Each Stream ──────────────────────────
        print(f"\n  ── Step 1: Normalizing Streams ──")

        dinov2_norm = self._l2_normalize(dinov2_raw)
        derm_norm   = self._l2_normalize(derm_raw)
        abcd_norm   = self._l2_normalize(self._minmax_normalize(abcd_raw))

        print(f"  DINOv2          : L2 norm  → shape {dinov2_norm.shape}"
              f"  magnitude {np.linalg.norm(dinov2_norm):.4f}")
        print(f"  Derm Foundation : L2 norm  → shape {derm_norm.shape}"
              f"  magnitude {np.linalg.norm(derm_norm):.4f}")
        print(f"  ABCD Scores     : MinMax+L2 → shape {abcd_norm.shape}"
              f"  range [{abcd_norm.min():.4f}, {abcd_norm.max():.4f}]")

        # ── Step 2: Equal Stream Weighting ─────────────────────────
        # Problem without weighting:
        #   Derm Foundation has 6144 dims vs DINOv2 768 dims vs ABCD 4 dims.
        #   Even after L2 norm, downstream models will implicitly favor
        #   Derm Foundation simply because it contributes 8x more values.
        # Fix:
        #   Multiply each normalized stream by 1/3 so all three streams
        #   contribute exactly equally to the combined feature vector.
        print(f"\n  ── Step 2: Equal Stream Weighting"
              f" (each × {STREAM_WEIGHT:.4f}) ──")

        dinov2_weighted = dinov2_norm * STREAM_WEIGHT
        derm_weighted   = derm_norm   * STREAM_WEIGHT
        abcd_weighted   = abcd_norm   * STREAM_WEIGHT

        print(f"  DINOv2          : weighted magnitude"
              f" {np.linalg.norm(dinov2_weighted):.4f}")
        print(f"  Derm Foundation : weighted magnitude"
              f" {np.linalg.norm(derm_weighted):.4f}")
        print(f"  ABCD Scores     : weighted magnitude"
              f" {np.linalg.norm(abcd_weighted):.4f}")

        # ── Step 3: Concatenate All Three Streams ──────────────────
        feature_vector = np.concatenate([
            dinov2_weighted,   # 768 dims  — dermoscopic features
            derm_weighted,     # 6144 dims — clinical features
            abcd_weighted      # 4 dims    — geometric clinical scores
        ], axis=0)             # total: 6916 dims

        feature_dim = feature_vector.shape[0]

        print(f"\n  ✓ Combined feature vector : {feature_vector.shape}")
        print(f"    Expected               : ({COMBINED_FEATURE_DIM},)")
        print(f"    Match                  : "
              f"{'YES' if feature_dim == COMBINED_FEATURE_DIM else 'NO — check config'}")
        print(f"{'═' * 60}\n")

        return {
            "status"        : "PASS",
            "reason"        : (
                "Feature extraction complete. "
                "L2 norm on embeddings, MinMax+L2 on ABCD. "
                "Equal 1/3 weighting on all three streams."
            ),
            "feature_vector": feature_vector,
            "feature_dim"   : feature_dim,
            "components"    : {
                "dinov2_raw"       : dinov2_raw,
                "dinov2_norm"      : dinov2_norm,
                "dinov2_weighted"  : dinov2_weighted,
                "derm_raw"         : derm_raw,
                "derm_norm"        : derm_norm,
                "derm_weighted"    : derm_weighted,
                "abcd_raw"         : abcd_raw,
                "abcd_norm"        : abcd_norm,
                "abcd_weighted"    : abcd_weighted,
                "stream_weight"    : STREAM_WEIGHT,
            }
        }

    # ─── Save Features to Disk ─────────────────────────────────────

    def save_features(
        self,
        feature_vector : np.ndarray,
        image_path     : str,
        output_dir     : str = "data/processed"
    ) -> str:
        """
        Save the combined feature vector to disk as a .npy file.
        Called optionally by the Orchestrator after run() succeeds.

        Returns:
            str: Path to the saved .npy file.
        """
        os.makedirs(output_dir, exist_ok=True)
        base_name   = os.path.splitext(
            os.path.basename(image_path))[0]
        output_path = os.path.join(
            output_dir, f"{base_name}_features.npy")
        np.save(output_path, feature_vector)
        print(f"  [Agent 2] Features saved to: {output_path}")
        return output_path