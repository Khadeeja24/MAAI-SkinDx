# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 4 — XAI Explainability Agent
# ══════════════════════════════════════════════════════════════════
# Runs AFTER Agent 3 finishes.
# Runs IN PARALLEL with Clinical Agent.
# Reports back to Orchestrator only.
#
# Two-part explanation system:
#   Part 1 — Grad-CAM: pure mathematics, hooks into layer4
#             produces heatmap and activation statistics
#   Part 2 — LLM: Groq LLaMA-3.3-70b-versatile generates
#             natural language clinical explanation from stats
#             Falls back to template if Groq unavailable
#
# This makes Agent 4 a genuine AI agent:
#   Perceives  : image + Agent 3 model + predictions
#   Processes  : Grad-CAM gradients + LLM reasoning
#   Acts       : produces heatmap + clinical explanation
#
# All production fixes applied:
#   - torch.set_grad_enabled(True) prevents silent zero gradients
#   - model.zero_grad() before forward clears stale gradients
#   - Original image resolution preserved in overlay
#   - Malignant class flagged in explanation
#   - CUDA memory cleanup after backward
#   - Robust focus detection with peak + spread statistics
#   - Secondary CAM for low confidence predictions
#   - UTF-8 encoding on all file writes
# ══════════════════════════════════════════════════════════════════

import os
import sys
import numpy as np
import torch
import cv2
import traceback
import warnings
warnings.filterwarnings("ignore")

from PIL import Image as PILImage
from torchvision import transforms

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import (
    OUTPUT_DIR, IMG_SIZE, ALPHA,
    CLASS_EXPLANATIONS, CLASS_URGENCY
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MALIGNANT_CLASSES        = {0, 2, 3}
SECONDARY_CAM_THRESHOLD  = 0.60
LLM_MODEL                = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS           = 400
LLM_TEMPERATURE          = 0.3

# Try to import Groq — fall back to template if not available
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]),
])


class XAIAgent:

    def __init__(self):
        self.name   = "Agent 4 — XAI Explainability Agent"
        self.client = None
        self._init_llm()
        print(f"\n[{self.name}] Initialised")
        if self.client:
            print(f"  [Agent 4] LLM : Groq {LLM_MODEL} ready")
        else:
            print(f"  [Agent 4] LLM : Not available — using templates")

    # ─── LLM Initialisation ────────────────────────────────────────

    def _init_llm(self):
        """
        Initialise Groq client.
        Requires GROQ_API_KEY environment variable.
        Fails silently — pipeline falls back to templates.
        """
        if not GROQ_AVAILABLE:
            return
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            return
        try:
            self.client = Groq(api_key=api_key)
        except Exception:
            self.client = None

    # ─── Grad-CAM Core ─────────────────────────────────────────────

    def _compute_gradcam(
        self,
        model,
        img_tensor   : torch.Tensor,
        a2_tensor,
        target_class : int
    ) -> np.ndarray:
        """
        Compute Grad-CAM for target class using layer4 hook.

        Critical implementation details:
        1. torch.set_grad_enabled(True) — explicit, never assumed.
           Any outer torch.no_grad() silently zeros all gradients
           without this. Backward completes with no error but
           produces a blank heatmap.
        2. model.zero_grad() BEFORE forward — clears stale gradients
           from previous cases that would corrupt current CAM.
        3. Activations and gradients in lists — correct closure
           capture pattern in Python, avoids key collision.
        4. Hooks always removed in finally regardless of failure.
        5. Explicit CUDA memory cleanup prevents VRAM accumulation.

        Returns normalized CAM (7,7). Returns zeros on failure.
        """
        activations = []
        gradients   = []

        def forward_hook(module, inp, out):
            activations.append(out.detach())

        def backward_hook(module, grad_inp, grad_out):
            gradients.append(grad_out[0].detach())

        handle_f = model.layer4.register_forward_hook(forward_hook)
        handle_b = model.layer4.register_full_backward_hook(
            backward_hook)

        try:
            torch.set_grad_enabled(True)
            model.eval()
            img_tensor = img_tensor.to(DEVICE)
            if a2_tensor is not None:
                a2_tensor = a2_tensor.to(DEVICE)

            model.zero_grad()
            output, _ = model(img_tensor, a2_tensor)

            score = output[0, target_class]
            score.backward()

            acts    = activations[0]
            grads   = gradients[0]
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam     = (weights * acts).sum(dim=1, keepdim=True)
            cam     = torch.relu(cam)
            cam     = cam.squeeze().cpu().numpy()

            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max - cam_min > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

        except Exception as e:
            print(f"  [Agent 4] Grad-CAM error: {e}")
            traceback.print_exc()
            cam = np.zeros((7, 7), dtype=np.float32)

        finally:
            handle_f.remove()
            handle_b.remove()
            model.zero_grad()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        return cam

    # ─── Visualization ─────────────────────────────────────────────

    def _cam_to_heatmap_rgb(self, cam: np.ndarray) -> np.ndarray:
        """Convert 7x7 CAM to 224x224 RGB heatmap (jet colormap)."""
        cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        cam_uint8   = np.uint8(255 * cam_resized)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    def _overlay_on_original(
        self,
        original_path : str,
        cam           : np.ndarray,
    ) -> np.ndarray:
        """
        Overlay heatmap at original image resolution.
        Previous version forced 224x224 — destroys clinical photo
        resolution. Resize CAM to match original dimensions exactly.
        """
        original       = PILImage.open(original_path).convert("RGB")
        orig_w, orig_h = original.size
        original_arr   = np.array(original)

        cam_resized = cv2.resize(cam, (orig_w, orig_h))
        cam_uint8   = np.uint8(255 * cam_resized)
        heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        return (ALPHA * heatmap_rgb +
                (1 - ALPHA) * original_arr).astype(np.uint8)

    # ─── Focus Detection ───────────────────────────────────────────

    def _analyze_focus(self, cam: np.ndarray) -> dict:
        """
        Robust focus detection using peak location and spread.
        Previous version only compared centre vs border mean.
        Fails when overall activation is low or when peak sits
        on the boundary zone.

        Uses: peak location + activation spread (std) for robust
        classification into lesion/border/diffuse focus types.
        """
        h, w     = cam.shape
        margin   = 2

        flat_idx = int(np.argmax(cam))
        peak_row = flat_idx // w
        peak_col = flat_idx % w
        max_val  = float(cam.max())
        spread   = float(cam.std())

        in_centre = (margin <= peak_row < h - margin and
                     margin <= peak_col < w - margin)

        if max_val < 0.25:
            focus_type = "focus_diffuse"
        elif spread < 0.12:
            focus_type = "focus_diffuse"
        elif in_centre:
            focus_type = "focus_on_lesion"
        else:
            focus_type = "focus_on_border"

        return {
            "focus_type" : focus_type,
            "peak_row"   : int(peak_row),
            "peak_col"   : int(peak_col),
            "spread"     : round(spread, 4),
            "max_val"    : round(max_val, 4),
        }

    # ─── LLM Explanation ───────────────────────────────────────────

    def _llm_explanation(
        self,
        predicted_index  : int,
        predicted_name   : str,
        confidence       : float,
        focus_stats      : dict,
        top_5            : list,
        is_malignant     : bool,
    ) -> str:
        """
        Generate clinical explanation using Groq LLaMA-3.3-70b.

        The LLM receives structured Grad-CAM statistics and
        prediction context. It generates a natural language
        clinical paragraph suitable for a dermatologist.

        System prompt enforces:
        - Professional clinical language
        - Specific reference to visual features detected
        - Malignancy urgency when applicable
        - 3-4 sentence limit for readability

        Returns LLM-generated text, or None if LLM unavailable.
        None causes fallback to template in _generate_explanation.
        """
        if self.client is None:
            return None

        focus_readable = focus_stats["focus_type"].replace(
            "_", " ").title()

        top5_text = "\n".join([
            f"  {p['rank']}. {p['class_name']} — "
            f"{p['confidence_pct']}"
            for p in (top_5 or [])[:5]
        ])

        malignant_instruction = (
            "IMPORTANT: This is a MALIGNANT class prediction. "
            "The explanation MUST include a clear urgent referral "
            "recommendation."
            if is_malignant else ""
        )

        system_prompt = (
            "You are a clinical AI explainability assistant "
            "specialising in dermatology. You explain AI skin "
            "disease predictions to qualified dermatologists. "
            "Your explanations are precise, clinically grounded, "
            "and free of unnecessary hedging. "
            "Write in professional medical language. "
            "Always be specific about what visual features the "
            "AI detected and why they are clinically relevant. "
            "Maximum 4 sentences."
        )

        user_prompt = (
            f"An AI system has analysed a skin lesion image.\n\n"
            f"PREDICTION:\n"
            f"  Condition   : {predicted_name}\n"
            f"  Confidence  : {confidence*100:.1f}%\n"
            f"  Is malignant: {'YES' if is_malignant else 'No'}\n\n"
            f"GRAD-CAM VISUAL ANALYSIS:\n"
            f"  Focus pattern      : {focus_readable}\n"
            f"  Peak activation    : row {focus_stats['peak_row']}, "
            f"col {focus_stats['peak_col']} (on 7x7 spatial grid)\n"
            f"  Activation spread  : {focus_stats['spread']:.4f} "
            f"(0=very focused, 0.5=diffuse)\n"
            f"  Max activation     : {focus_stats['max_val']:.4f}\n\n"
            f"TOP 5 PREDICTIONS:\n{top5_text}\n\n"
            f"{malignant_instruction}\n\n"
            f"Write a 3-4 sentence clinical explanation of what "
            f"visual features the AI focused on, why they are "
            f"relevant to the predicted condition, and what "
            f"clinical action this suggests."
        )

        try:
            response = self.client.chat.completions.create(
                model       = LLM_MODEL,
                max_tokens  = LLM_MAX_TOKENS,
                temperature = LLM_TEMPERATURE,
                messages    = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ]
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"  [Agent 4] LLM error: {e} — using template")
            return None

    def _template_explanation(
        self,
        predicted_index : int,
        predicted_name  : str,
        focus_type      : str,
    ) -> str:
        """
        Template-based explanation fallback when LLM unavailable.
        Uses pre-written clinical text from config.py.
        """
        exps = CLASS_EXPLANATIONS.get(predicted_index, {})
        return exps.get(
            focus_type,
            exps.get(
                "focus_diffuse",
                f"The model identified features consistent "
                f"with {predicted_name}."
            )
        )

    # ─── Full Explanation Generation ───────────────────────────────

    def _generate_explanation(
        self,
        predicted_index  : int,
        predicted_name   : str,
        confidence       : float,
        focus_stats      : dict,
        top_5            : list,
        case_id          : str,
        secondary_index  : int   = None,
        secondary_name   : str   = None,
        secondary_conf   : float = None,
    ) -> str:
        """
        Generate complete written explanation.

        Structure:
          PREDICTION SUMMARY  — always template-based (factual)
          VISUAL EXPLANATION  — LLM if available, else template
          MALIGNANT WARNING   — template (safety-critical, not LLM)
          HEATMAP STATS       — always template-based (factual)
          SECONDARY CAM NOTE  — template (factual)
          IMPORTANT NOTICE    — always template (legal/safety)
        """
        is_malignant = predicted_index in MALIGNANT_CLASSES
        urgency      = CLASS_URGENCY.get(
            predicted_index, "Clinical assessment recommended.")
        focus_type   = focus_stats["focus_type"]

        # Try LLM first — fall back to template
        visual_exp = self._llm_explanation(
            predicted_index = predicted_index,
            predicted_name  = predicted_name,
            confidence      = confidence,
            focus_stats     = focus_stats,
            top_5           = top_5,
            is_malignant    = is_malignant,
        )
        llm_used = visual_exp is not None
        if not llm_used:
            visual_exp = self._template_explanation(
                predicted_index, predicted_name, focus_type)

        # Malignant safety block — always template, not LLM
        # Safety-critical text must be deterministic
        malignant_block = ""
        if is_malignant:
            malignant_block = (
                "\n"
                "  *** MALIGNANT CLASS PREDICTED ***\n"
                "  This prediction falls into a malignant category.\n"
                "  Clinical evaluation is strongly recommended\n"
                "  regardless of the confidence score shown.\n"
                "  Do not dismiss based on AI confidence alone.\n"
            )

        # Secondary CAM block
        secondary_block = ""
        if secondary_index is not None:
            secondary_block = (
                f"\nSECONDARY PREDICTION (Low Confidence Case)\n"
                f"  Second class       : {secondary_name}\n"
                f"  Second confidence  : {secondary_conf*100:.1f}%\n"
                f"  A secondary heatmap has been saved showing\n"
                f"  which regions support the alternative diagnosis.\n"
                f"  Review both heatmaps together.\n"
            )

        explanation_source = (
            "LLM-generated (Groq LLaMA-3.3-70b)"
            if llm_used else
            "Template-based (Groq unavailable)"
        )

        return (
            f"MAAI-SkinDx | Agent 4 \u2014 XAI Explanation\n"
            f"Case ID            : {case_id}\n"
            f"Explanation source : {explanation_source}\n"
            f"{'='*60}\n\n"
            f"PREDICTION SUMMARY\n"
            f"  Predicted condition : {predicted_name}\n"
            f"  Confidence          : {confidence*100:.1f}%\n"
            f"  Clinical urgency    : {urgency}\n"
            f"{malignant_block}\n"
            f"VISUAL EXPLANATION\n"
            f"  {visual_exp}\n"
            f"\n"
            f"HEATMAP INTERPRETATION\n"
            f"  Red / warm regions  : high importance to prediction\n"
            f"  Blue / cool regions : low importance\n\n"
            f"  Focus pattern     : "
            f"{focus_type.replace('_', ' ').title()}\n"
            f"  Peak location     : "
            f"row {focus_stats['peak_row']}, "
            f"col {focus_stats['peak_col']} (7x7 grid)\n"
            f"  Activation spread : {focus_stats['spread']:.4f}\n"
            f"  Max activation    : {focus_stats['max_val']:.4f}\n"
            f"{secondary_block}\n"
            f"IMPORTANT NOTICE\n"
            f"  This AI explanation supports clinical decision\n"
            f"  making. It does not replace it. A clinician must\n"
            f"  verify the highlighted regions align with their\n"
            f"  own clinical assessment before acting.\n\n"
            f"{'='*60}\n"
        )

    # ─── Main Entry Point ──────────────────────────────────────────

    def run(
        self,
        image_path      : str,
        agent3_model,
        agent2_features,
        predicted_index : int,
        predicted_name  : str,
        confidence      : float,
        use_agent2      : bool,
        case_id         : str,
        top_5           : list = None,
    ) -> dict:
        """
        Generate Grad-CAM explanation for Agent 3 prediction.
        Called by Orchestrator after Agent 3 finishes.
        Runs in parallel with Clinical Agent.
        """
        print(f"\n{'='*60}")
        print(f"  {self.name}")
        print(f"  Case      : {case_id}")
        print(f"  Explaining: {predicted_name} "
              f"({confidence*100:.1f}%)")
        print(f"  LLM       : "
              f"{'Groq active' if self.client else 'template fallback'}")
        if predicted_index in MALIGNANT_CLASSES:
            print(f"  WARNING   : Malignant class — clinical flag active")
        print(f"{'='*60}")

        if agent3_model is None:
            return {
                "status"          : "FAIL",
                "reason"          : "Agent 3 model not provided.",
                "heatmap_path"    : None,
                "overlay_path"    : None,
                "secondary_path"  : None,
                "explanation"     : None,
                "explanation_path": None,
                "cam"             : None,
                "focus_stats"     : None,
            }

        try:
            img    = PILImage.open(image_path).convert("RGB")
            tensor = VAL_TRANSFORM(img).unsqueeze(0)

            a2_tensor = None
            if use_agent2 and agent2_features is not None:
                a2_np = np.asarray(agent2_features, dtype=np.float32)
                if a2_np.shape[0] == 6916:
                    a2_tensor = torch.from_numpy(
                        a2_np).unsqueeze(0)

            # Primary Grad-CAM
            print(f"\n  -- Computing Primary Grad-CAM --")
            print(f"  Target: class {predicted_index} "
                  f"({predicted_name})")
            cam = self._compute_gradcam(
                agent3_model, tensor, a2_tensor, predicted_index)
            print(f"  CAM shape  : {cam.shape}")
            print(f"  CAM range  : [{cam.min():.3f}, {cam.max():.3f}]")

            focus_stats = self._analyze_focus(cam)
            print(f"  Focus type : {focus_stats['focus_type']}")
            print(f"  Peak       : row={focus_stats['peak_row']} "
                  f"col={focus_stats['peak_col']}")
            print(f"  Spread     : {focus_stats['spread']:.4f}")

            # Secondary Grad-CAM for low confidence
            secondary_cam   = None
            secondary_index = None
            secondary_name  = None
            secondary_conf  = None
            secondary_path  = None

            if (confidence < SECONDARY_CAM_THRESHOLD and
                    top_5 and len(top_5) >= 2):
                secondary_index = top_5[1].get("class_index")
                secondary_name  = top_5[1].get("class_name", "")
                secondary_conf  = top_5[1].get("confidence", 0.0)

                if secondary_index is not None:
                    print(f"\n  -- Computing Secondary Grad-CAM --")
                    print(f"  Confidence {confidence*100:.1f}% < "
                          f"{SECONDARY_CAM_THRESHOLD*100:.0f}%")
                    print(f"  Secondary: {secondary_name} "
                          f"({secondary_conf*100:.1f}%)")
                    secondary_cam = self._compute_gradcam(
                        agent3_model, tensor,
                        a2_tensor, secondary_index)

            # LLM explanation
            print(f"\n  -- Generating LLM Explanation --")
            explanation = self._generate_explanation(
                predicted_index = predicted_index,
                predicted_name  = predicted_name,
                confidence      = confidence,
                focus_stats     = focus_stats,
                top_5           = top_5,
                case_id         = case_id,
                secondary_index = secondary_index,
                secondary_name  = secondary_name,
                secondary_conf  = secondary_conf,
            )

            # Generate and save images
            heatmap_std  = self._cam_to_heatmap_rgb(cam)
            overlay_orig = self._overlay_on_original(image_path, cam)

            os.makedirs(OUTPUT_DIR, exist_ok=True)

            heatmap_path     = os.path.join(
                OUTPUT_DIR, f"{case_id}_gradcam_heatmap.png")
            overlay_path     = os.path.join(
                OUTPUT_DIR, f"{case_id}_gradcam_overlay.png")
            explanation_path = os.path.join(
                OUTPUT_DIR, f"{case_id}_explanation.txt")

            PILImage.fromarray(heatmap_std).save(heatmap_path)
            PILImage.fromarray(overlay_orig).save(overlay_path)
            with open(explanation_path, "w", encoding="utf-8") as f:
                f.write(explanation)

            if secondary_cam is not None:
                secondary_path = os.path.join(
                    OUTPUT_DIR, f"{case_id}_gradcam_secondary.png")
                sec_heatmap = self._cam_to_heatmap_rgb(secondary_cam)
                sec_overlay = self._overlay_on_original(
                    image_path, secondary_cam)
                sec_h = min(sec_heatmap.shape[0],
                            sec_overlay.shape[0])
                composite = np.concatenate(
                    [sec_heatmap[:sec_h], sec_overlay[:sec_h]],
                    axis=1)
                PILImage.fromarray(composite).save(secondary_path)
                print(f"  Secondary: {secondary_path}")

            print(f"\n  Heatmap  : {heatmap_path}")
            print(f"  Overlay  : {overlay_path}")
            print(f"  Explained: {explanation_path}")
            print(f"{'='*60}\n")

            return {
                "status"          : "PASS",
                "reason"          : "Grad-CAM explanation generated.",
                "heatmap_path"    : heatmap_path,
                "overlay_path"    : overlay_path,
                "secondary_path"  : secondary_path,
                "explanation"     : explanation,
                "explanation_path": explanation_path,
                "cam"             : cam,
                "focus_stats"     : focus_stats,
                "secondary_cam"   : secondary_cam,
                "secondary_class" : secondary_name,
            }

        except Exception as e:
            print(f"  [Agent 4] Error: {e}")
            traceback.print_exc()
            return {
                "status"          : "FAIL",
                "reason"          : str(e),
                "heatmap_path"    : None,
                "overlay_path"    : None,
                "secondary_path"  : None,
                "explanation"     : None,
                "explanation_path": None,
                "cam"             : None,
                "focus_stats"     : None,
            }