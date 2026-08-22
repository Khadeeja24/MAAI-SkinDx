# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Orchestrator
# ══════════════════════════════════════════════════════════════════
# Central manager of the entire MAAI-SkinDx pipeline.
# All agents report to the Orchestrator only.
# No agent talks directly to another agent.
#
# Pipeline flow:
#   Patient submits image + symptoms
#   Orchestrator → Agent 1 (image quality check)
#   Orchestrator → Agent 2 (visual feature extraction)
#   Orchestrator → Agent 3 (disease classification)
#   Agent 3 finishes →
#     Agent 4 (XAI Grad-CAM) + Clinical Agent run IN PARALLEL
#   Both finish → Agent 5 (RAG) — coming next
#   Agent 5 finishes → Orchestrator 5-stage review
#   Review done → Agent 7 (clinical report) — coming next
#   Agent 6 → Background learning (completely separate)
# ══════════════════════════════════════════════════════════════════

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from agents.agent1_image_quality      import ImageQualityAgent
from agents.agent2_feature_extraction import FeatureExtractionAgent
from agents.agent3_diagnosis          import DiagnosisAgent
from agents.agent4_xai                import XAIAgent
from agents.clinical_agent            import ClinicalAgent


class Orchestrator:
    """
    Central coordinator for the MAAI-SkinDx pipeline.
    Receives patient submissions and manages agent execution.
    No agent ever talks to another agent directly.
    """

    def __init__(self):
        print("\n" + "═" * 60)
        print("  MAAI-SkinDx | Orchestrator Initialising")
        print("═" * 60)

        print("\n  Loading agents...")
        self.agent1   = ImageQualityAgent()
        self.agent2   = FeatureExtractionAgent()
        self.agent3   = DiagnosisAgent()
        self.agent4   = XAIAgent()
        self.clinical = ClinicalAgent()
        print("  All agents ready.")
        print("═" * 60 + "\n")

    def process(self, image_path: str, symptoms: dict = None) -> dict:
        """
        Main entry point. Patient submits image + symptoms.
        Orchestrator manages the full pipeline.

        Args:
            image_path (str)  : Path to the uploaded image.
            symptoms   (dict) : Patient symptom data from form.
                                Keys: age, itch, grew, hurt,
                                      changed, bleed, elevation,
                                      skin_cancer_history,
                                      cancer_history, fitspatrick,
                                      gender, region, diameter_1,
                                      diameter_2, smoke, drink

        Returns:
            dict: Full pipeline result with all agent outputs.
        """
        start_time = time.time()
        case_id    = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("\n" + "═" * 60)
        print(f"  MAAI-SkinDx | New Case: {case_id}")
        print(f"  Image  : {Path(image_path).name}")
        print(f"  Symptoms provided: {'Yes' if symptoms else 'No'}")
        print("═" * 60)

        # ── Result dictionary ──────────────────────────────────────
        result = {
            "case_id"        : case_id,
            "image_path"     : image_path,
            "symptoms"       : symptoms or {},
            "pipeline_status": "RUNNING",
            "agents"         : {},
            "final_status"   : None,
            "message"        : None,
            "elapsed_seconds": None,
        }

        # ══════════════════════════════════════════════════════════
        # STEP 1 — Agent 1: Image Quality Check
        # ══════════════════════════════════════════════════════════
        print("\n── Step 1: Agent 1 — Image Quality Check ──")

        agent1_result = self.agent1.run(image_path)
        result["agents"]["agent1"] = agent1_result

        if agent1_result["status"] == "FAIL":
            elapsed = round(time.time() - start_time, 2)
            result.update({
                "pipeline_status": "REJECTED",
                "final_status"   : "FAIL",
                "message"        : agent1_result["reason"],
                "elapsed_seconds": elapsed,
            })
            self._print_summary(result)
            return result

        print(f"  Agent 1 → PASS. Image approved.")

        # ══════════════════════════════════════════════════════════
        # STEP 2 — Agent 2: Visual Feature Extraction
        # ══════════════════════════════════════════════════════════
        print("\n── Step 2: Agent 2 — Feature Extraction ──")

        agent2_result = self.agent2.run(image_path)
        result["agents"]["agent2"] = agent2_result

        if agent2_result["status"] == "FAIL":
            elapsed = round(time.time() - start_time, 2)
            result.update({
                "pipeline_status": "ERROR",
                "final_status"   : "FAIL",
                "message"        : agent2_result.get(
                    "reason", "Feature extraction failed."),
                "elapsed_seconds": elapsed,
            })
            self._print_summary(result)
            return result

        feature_vector = agent2_result.get("feature_vector")
        feature_dim    = agent2_result.get("feature_dim", 0)
        print(f"\n  Agent 2 → PASS. "
              f"Features extracted: {feature_dim} dimensions.")

        # ══════════════════════════════════════════════════════════
        # STEP 3 — Agent 3: Disease Classification
        # ══════════════════════════════════════════════════════════
        print("\n── Step 3: Agent 3 — Disease Classification ──")

        agent3_result = self.agent3.run(
            image_path      = image_path,
            agent2_features = feature_vector
        )
        result["agents"]["agent3"] = agent3_result

        if agent3_result["status"] == "FAIL":
            elapsed = round(time.time() - start_time, 2)
            result.update({
                "pipeline_status": "ERROR",
                "final_status"   : "FAIL",
                "message"        : agent3_result.get(
                    "reason", "Classification failed."),
                "elapsed_seconds": elapsed,
            })
            self._print_summary(result)
            return result

        predicted_name  = agent3_result.get("predicted_name",  "Unknown")
        predicted_class = agent3_result.get("predicted_class", "Unknown")
        predicted_index = agent3_result.get("predicted_index", 0)
        confidence      = agent3_result.get("confidence", 0) * 100

        print(f"\n  Agent 3 → PASS.")
        print(f"  Predicted disease : {predicted_name}")
        print(f"  Confidence        : {confidence:.1f}%")

        # ══════════════════════════════════════════════════════════
        # STEPS 4 + 5 — Agent 4 (XAI) + Clinical Agent — PARALLEL
        #
        # Agent 3 has finished its image-based diagnosis.
        # Now two agents start at the same time:
        #
        # Agent 4 — XAI Explainability:
        #   Uses Grad-CAM to explain WHY Agent 3 made that
        #   prediction. Generates heatmap overlay and written
        #   explanation for the doctor.
        #
        # Clinical Agent — Symptom Risk Assessment:
        #   Takes patient form data (age, symptoms, history)
        #   and produces a clinical risk score independently
        #   of the image diagnosis.
        #
        # These two do not depend on each other → run in parallel.
        # Both report results back to Orchestrator when done.
        # ══════════════════════════════════════════════════════════
        print("\n── Steps 4+5: Agent 4 (XAI) + Clinical Agent [parallel] ──")

        agent4_result   = None
        clinical_result = None

        def run_agent4():
            """Agent 4 — Grad-CAM XAI explainability."""
            return self.agent4.run(
                image_path      = image_path,
                agent3_model    = self.agent3.model,
                agent2_features = feature_vector,
                predicted_index = predicted_index,
                predicted_name  = predicted_class,
                confidence      = agent3_result.get("confidence", 0.0),
                use_agent2      = self.agent3.use_agent2,
                case_id         = case_id,
                top_5           = agent3_result.get("top_5_predictions", []),
            )

        def run_clinical():
            """Clinical Agent — patient symptom risk scoring."""
            return self.clinical.run(
                symptoms = symptoms or {},
                case_id  = case_id,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(run_agent4)   : "agent4",
                executor.submit(run_clinical) : "clinical",
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    res = future.result()
                    if name == "agent4":
                        agent4_result   = res
                    else:
                        clinical_result = res
                    print(f"  {name.upper()} completed — "
                          f"status: {res.get('status', 'UNKNOWN')}")
                except Exception as e:
                    print(f"  {name.upper()} failed: {e}")
                    if name == "agent4":
                        agent4_result   = {
                            "status": "FAIL", "reason": str(e)}
                    else:
                        clinical_result = {
                            "status": "FAIL", "reason": str(e)}

        result["agents"]["agent4"]   = agent4_result
        result["agents"]["clinical"] = clinical_result

        # ══════════════════════════════════════════════════════════
        # STEPS 6, 7, 8 — Coming next
        #
        # Agent 5 : RAG knowledge retrieval
        #   Searches confirmed past cases matching image findings
        #   + clinical symptoms combined
        #
        # Orchestrator 5-stage review:
        #   Resolves disagreements between image diagnosis (Agent 3),
        #   XAI explanation (Agent 4), and clinical risk (Clinical)
        #   to produce one final trustworthy diagnosis
        #
        # Agent 7 : Clinical report generation
        #   Writes official clinical report in medical format
        #   ready for a doctor to read
        #
        # Agent 6 : Background continuous learning
        #   Completely separate — never blocks this pipeline
        # ══════════════════════════════════════════════════════════

        # ── Build final result ──────────────────────────────────────
        elapsed     = round(time.time() - start_time, 2)
        xai_status  = agent4_result.get("status", "FAIL") \
                      if agent4_result else "FAIL"
        overlay     = agent4_result.get("overlay_path", "") \
                      if agent4_result else ""
        explanation = agent4_result.get("explanation", "") \
                      if agent4_result else ""
        clin_status = clinical_result.get("status", "FAIL") \
                      if clinical_result else "FAIL"
        risk_level  = clinical_result.get("risk_level", None) \
                      if clinical_result else None

        # Build message
        clin_msg = ""
        if risk_level:
            clin_msg = f" Clinical risk: {risk_level}."

        result.update({
            "pipeline_status"  : "PARTIAL",
            "final_status"     : "PASS",
            "predicted_disease": predicted_name,
            "confidence"       : f"{confidence:.1f}%",
            "top_5_predictions": agent3_result.get(
                "top_5_predictions", []),
            "xai_overlay"      : overlay,
            "xai_status"       : xai_status,
            "xai_explanation"  : explanation,
            "clinical_status"  : clin_status,
            "clinical_risk"    : risk_level,
            "message"          : (
                f"Image quality check passed. "
                f"Features extracted ({feature_dim} dims). "
                f"Disease classified: {predicted_name} "
                f"({confidence:.1f}% confidence). "
                f"XAI explanation generated.{clin_msg} "
                f"RAG and report generation coming next."
            ),
            "elapsed_seconds"  : elapsed,
        })

        self._print_summary(result)
        return result

    def _print_summary(self, result: dict):
        """Print clean pipeline summary."""
        print("\n" + "═" * 60)
        print(f"  Pipeline Summary — Case: {result['case_id']}")
        print("═" * 60)
        print(f"  Status  : {result['pipeline_status']}")
        print(f"  Message : {result['message']}")
        print(f"  Time    : {result['elapsed_seconds']}s")

        for agent_name, agent_result in result["agents"].items():
            status = agent_result.get("status", "UNKNOWN")
            print(f"  {agent_name.upper()}: {status}")

        if result.get("top_5_predictions"):
            print(f"\n  Top 5 Disease Predictions:")
            for pred in result["top_5_predictions"]:
                bar = '█' * int(pred['confidence'] * 20)
                print(f"    {pred['rank']}. "
                      f"{pred['class_name']:<35} "
                      f"{pred['confidence_pct']:>6}  {bar}")

        if result.get("xai_overlay"):
            print(f"\n  XAI Overlay : {result['xai_overlay']}")
            print(f"  XAI Status  : {result.get('xai_status', '')}")

        clin = result.get("agents", {}).get("clinical", {})
        if clin and clin.get("risk_level"):
            print(f"\n  Clinical Risk  : {clin['risk_level']} "
                  f"(score {clin['risk_score']}/100)")
            if clin.get("ml_probability") is not None:
                print(f"  ML Probability : "
                      f"{clin['ml_probability']*100:.1f}%")
            print(f"  Recommendation : {clin['recommendation']}")

        print("═" * 60 + "\n")