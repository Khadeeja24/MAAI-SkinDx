# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Orchestrator
# ══════════════════════════════════════════════════════════════════
# Central manager of the entire MAAI-SkinDx pipeline.
# All agents report to the Orchestrator only.
# No agent talks directly to another agent.
#
# Current flow (sequential):
#   Patient submits image + symptoms
#   Orchestrator → Agent 1 (image quality check)
#   Orchestrator → Agent 2 (visual feature extraction)
#   Orchestrator → Agent 3 (disease classification)
#   [Agent 4, Clinical Agent, Agent 5, Agent 7 — coming next]
#
# Agent 3 inputs:
#   image_path      : approved image path from Agent 1
#   agent2_features : 6916-dim feature vector from Agent 2
#
# Agent 3 outputs:
#   top_5_predictions  : ranked disease predictions with confidence
#   backbone_features  : 2048-dim ResNet-50 features for Agent 4
# ══════════════════════════════════════════════════════════════════

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from agents.agent1_image_quality    import ImageQualityAgent
from agents.agent2_feature_extraction import FeatureExtractionAgent
from agents.agent3_diagnosis        import DiagnosisAgent


class Orchestrator:
    """
    Central coordinator for the MAAI-SkinDx pipeline.
    Receives patient submissions and manages agent execution.
    """

    def __init__(self):
        print("\n" + "═" * 60)
        print("  MAAI-SkinDx | Orchestrator Initialising")
        print("═" * 60)

        print("\n  Loading agents...")
        self.agent1 = ImageQualityAgent()
        self.agent2 = FeatureExtractionAgent()
        self.agent3 = DiagnosisAgent()
        print("  All agents ready.")
        print("═" * 60 + "\n")

    def process(self, image_path: str, symptoms: dict = None) -> dict:
        """
        Main entry point. Patient submits image + symptoms.
        Orchestrator manages the full pipeline.

        Args:
            image_path (str)  : Path to the uploaded image.
            symptoms   (dict) : Patient symptom data (optional).
                                Keys: age, lesion_duration, skin_tone,
                                      family_history, lesion_changed

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

        # ── Result dictionary ────────────────────────────────────
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

        print(f"\n  Status : PASS")
        print(f"  Reason : {agent1_result['reason']}")
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
            image_path     = image_path,
            agent2_features= feature_vector
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

        predicted_name = agent3_result.get("predicted_name", "Unknown")
        confidence     = agent3_result.get("confidence", 0) * 100
        print(f"\n  Agent 3 → PASS.")
        print(f"  Predicted disease : {predicted_name}")
        print(f"  Confidence        : {confidence:.1f}%")

        # ══════════════════════════════════════════════════════════
        # STEPS 4-7 — Placeholder (coming next)
        # ══════════════════════════════════════════════════════════
        # Agent 4 (XAI — Grad-CAM + SHAP) runs in parallel with:
        # Clinical Agent (symptom risk scoring)
        # Then:
        # Agent 5 (RAG knowledge retrieval)
        # Agent 7 (clinical report generation)
        # ══════════════════════════════════════════════════════════

        elapsed = round(time.time() - start_time, 2)
        result.update({
            "pipeline_status"  : "PARTIAL",
            "final_status"     : "PASS",
            "predicted_disease": predicted_name,
            "confidence"       : f"{confidence:.1f}%",
            "top_5_predictions": agent3_result.get(
                "top_5_predictions", []),
            "message"          : (
                f"Image quality check passed. "
                f"Features extracted ({feature_dim} dims). "
                f"Disease classified: {predicted_name} "
                f"({confidence:.1f}% confidence). "
                f"XAI, RAG, and report generation coming next."
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

        print("═" * 60 + "\n")