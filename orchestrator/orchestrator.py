# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Orchestrator — Adaptive Coordination Intelligence
# ══════════════════════════════════════════════════════════════════
# Central manager of the entire MAAI-SkinDx pipeline.
# All agents report to the Orchestrator only.
# No agent talks directly to another agent.
#
# Phase 1 improvements added:
#
#   Improvement 1 — OOD Detection
#     After Agent 3, confidence is checked against a threshold.
#     If max confidence < 0.40 across all 9 classes, the image
#     is flagged as out-of-distribution. Pipeline returns an
#     OOD warning instead of a potentially wrong diagnosis.
#
#   Improvement 2 — Dynamic Trust Scores
#     Each agent has a trust score derived from its validated
#     performance metrics. These scores are used in pre-fusion
#     reasoning to weight each agent's evidence appropriately.
#
#   Improvement 3 — Pre-Fusion Reasoning
#     Before combining Agent 3, Agent 4, and Clinical Agent
#     outputs, the Orchestrator assesses each agent's reliability,
#     confidence, and data quality. This produces a reasoned
#     assessment before fusion rather than blindly combining.
#
#   Improvement 4 — Conflict Detection Algorithm
#     Formally detects and classifies disagreements between
#     image-based diagnosis (Agent 3) and symptom-based risk
#     (Clinical Agent). Four conflict types defined.
#
#   Improvement 5 — Confidence-Aware Decision Policy
#     Replaces basic PASS/FAIL with a clinical decision:
#     CONCORDANT / ESCALATE / REVIEW / OOD / UNCERTAIN
#     Based on conflict type, confidence, and risk level.
#
# Pipeline flow:
#   Patient submits image + symptoms
#   Orchestrator → Agent 1 (image quality check)
#   Orchestrator → Agent 2 (visual feature extraction)
#   Orchestrator → Agent 3 (disease classification)
#   OOD check → flag or continue
#   Pre-fusion reasoning → assess each agent
#   Agent 4 + Clinical Agent [parallel]
#   Conflict detection → classify disagreement
#   Agent 5 (RAG knowledge retrieval)
#   Post-fusion reasoning → final decision
#   Agent 7 (clinical report) — coming next
#   Agent 6 (background learning) — separate
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
from agents.agent5_rag                import RAGAgent

# ── Improvement 1 — OOD threshold ──────────────────────────────────
# If Agent 3 max confidence below this, image is likely
# out-of-distribution (e.g. chickenpox, monkeypox).
# Based on empirical testing: in-distribution cases score
# 67-73%, OOD cases score 35-40%.
OOD_CONFIDENCE_THRESHOLD = 0.40

# ── Improvement 2 — Agent trust scores ─────────────────────────────
# Derived from validated performance metrics.
# Agent 1 : F1 97.89% on skin detection
# Agent 2 : 0 failures across all test cases
# Agent 3 : F1 74.18% on 7,734 hold-out images
# Agent 4 : qualitative — moderate trust
# Clinical: F1 92.68% on PAD-UFES-20 (when symptoms present)
# Agent 5 : RAG retrieval — knowledge quality moderate
AGENT_TRUST_SCORES = {
    "agent1"  : 0.979,   # based on test F1 0.9789
    "agent2"  : 0.950,   # feature extraction, 0 failures
    "agent3"  : 0.742,   # based on hold-out macro F1 74.18%
    "agent4"  : 0.850,   # XAI qualitative — moderate trust
    "clinical": 0.927,   # based on CV F1 92.68%
    "agent5"  : 0.800,   # RAG retrieval — moderate trust
}

# Malignant classes in Agent 3
MALIGNANT_CLASSES = {0, 2, 3}


class Orchestrator:
    """
    Adaptive Coordination Intelligence for MAAI-SkinDx.
    Implements pre-fusion reasoning, conflict detection,
    OOD detection, and confidence-aware decision making.
    No agent ever talks to another agent directly.
    """

    def __init__(self):
        print("\n" + "═" * 60)
        print("  MAAI-SkinDx | Orchestrator Initialising")
        print("  Mode: Adaptive Coordination Intelligence")
        print("═" * 60)

        print("\n  Loading agents...")
        self.agent1   = ImageQualityAgent()
        self.agent2   = FeatureExtractionAgent()
        self.agent3   = DiagnosisAgent()
        self.agent4   = XAIAgent()
        self.clinical = ClinicalAgent()
        self.agent5   = RAGAgent()
        print("  All agents ready.")
        print(f"  Trust scores: "
              f"A1={AGENT_TRUST_SCORES['agent1']:.3f} "
              f"A2={AGENT_TRUST_SCORES['agent2']:.3f} "
              f"A3={AGENT_TRUST_SCORES['agent3']:.3f} "
              f"Clinical={AGENT_TRUST_SCORES['clinical']:.3f}")
        print("═" * 60 + "\n")

    # ── Improvement 1 — OOD Detection ──────────────────────────────

    def _check_ood(self, confidence: float) -> bool:
        """
        Detect out-of-distribution images.

        When a patient submits an image of chickenpox, monkeypox,
        or any condition outside the 9 trained classes, Agent 3
        is forced to predict one of 9 classes regardless. The
        max confidence drops below 40% because no single class
        strongly activates.

        In-distribution: confidence 65-75%
        Out-of-distribution: confidence 35-40%

        Returns True if image is likely OOD.
        """
        return confidence < OOD_CONFIDENCE_THRESHOLD

    # ── Improvement 3 — Pre-Fusion Reasoning ───────────────────────

    def _pre_fusion_reasoning(
        self,
        agent3_result   : dict,
        confidence      : float,
        predicted_index : int,
        symptoms        : dict,
    ) -> dict:
        """
        Assess each agent's reliability before combining evidence.

        Pre-fusion reasoning evaluates:
        - Agent 3 confidence level (HIGH / MODERATE / LOW)
        - Whether malignant class is predicted
        - Whether clinical symptoms were provided
        - Combined evidence strength from both modalities

        Returns assessment dict used in decision making.
        """
        assessment = {}

        # Agent 3 confidence assessment
        if confidence >= 0.70:
            assessment["image_confidence_level"] = "HIGH"
        elif confidence >= 0.50:
            assessment["image_confidence_level"] = "MODERATE"
        else:
            assessment["image_confidence_level"] = "LOW"

        # Malignancy flag from image
        assessment["image_says_malignant"] = (
            predicted_index in MALIGNANT_CLASSES)

        # Clinical data availability
        assessment["clinical_data_available"] = (
            symptoms is not None and len(symptoms) > 0)

        # Agent 3 trust-weighted confidence
        assessment["weighted_image_confidence"] = (
            confidence * AGENT_TRUST_SCORES["agent3"])

        # Top prediction entropy — high entropy = uncertain
        top_5 = agent3_result.get("top_5_predictions", [])
        if top_5:
            import math
            probs = [p.get("confidence", 0) for p in top_5]
            entropy = -sum(
                p * math.log(p + 1e-10) for p in probs if p > 0)
            assessment["prediction_entropy"] = round(entropy, 4)
            assessment["high_entropy"] = entropy > 1.2
        else:
            assessment["prediction_entropy"] = 0.0
            assessment["high_entropy"] = False

        print(f"\n  -- Pre-Fusion Reasoning --")
        print(f"  Image confidence  : "
              f"{assessment['image_confidence_level']} "
              f"({confidence*100:.1f}%)")
        print(f"  Malignant class   : "
              f"{assessment['image_says_malignant']}")
        print(f"  Clinical data     : "
              f"{assessment['clinical_data_available']}")
        print(f"  Prediction entropy: "
              f"{assessment['prediction_entropy']:.4f} "
              f"({'HIGH' if assessment['high_entropy'] else 'NORMAL'})")
        print(f"  Weighted conf     : "
              f"{assessment['weighted_image_confidence']:.3f}")

        return assessment

    # ── Improvement 4 — Conflict Detection Algorithm ───────────────

    def _detect_conflict(
        self,
        predicted_index : int,
        confidence      : float,
        clinical_result : dict,
        assessment      : dict,
    ) -> dict:
        """
        Formally detect and classify disagreement between
        image-based diagnosis (Agent 3) and symptom-based
        risk assessment (Clinical Agent).

        Four conflict types:
        1. IMAGE_LOW_SYMPTOM_HIGH
           Image says benign, symptoms say malignant risk
           Most dangerous — must escalate
        2. IMAGE_HIGH_SYMPTOM_LOW
           Image says malignant, symptoms say low risk
           Requires review — image may be correct
        3. LOW_CONFIDENCE
           Agent 3 confidence too low to rely on image alone
           Need more evidence
        4. CONCORDANT
           Image and symptoms agree — proceed with confidence
        """
        conflict = {
            "type"        : "CONCORDANT",
            "severity"    : "NONE",
            "description" : "Image and clinical findings agree.",
        }

        if not clinical_result or clinical_result.get(
                "status") == "SKIP":
            conflict["type"] = "NO_CLINICAL_DATA"
            conflict["description"] = (
                "No symptom data provided. "
                "Decision based on image only.")
            return conflict

        ml_prob    = clinical_result.get("ml_probability")
        risk_level = clinical_result.get("risk_level", "LOW")
        risk_score = clinical_result.get("risk_score", 0)

        image_malignant   = predicted_index in MALIGNANT_CLASSES
        symptom_malignant = (
            ml_prob is not None and ml_prob > 0.60
        ) or risk_level in ("HIGH", "CRITICAL")

        if not image_malignant and symptom_malignant:
            conflict["type"]        = "IMAGE_LOW_SYMPTOM_HIGH"
            conflict["severity"]    = "HIGH"
            conflict["description"] = (
                f"Image predicts benign condition "
                f"({confidence*100:.1f}% confidence) but "
                f"clinical symptoms indicate malignancy risk "
                f"(ML {ml_prob*100:.1f}% if ml_prob else '',"
                f"risk score {risk_score}/100, {risk_level}). "
                f"Clinical symptoms must override — escalate."
            )

        elif image_malignant and not symptom_malignant:
            conflict["type"]        = "IMAGE_HIGH_SYMPTOM_LOW"
            conflict["severity"]    = "MODERATE"
            conflict["description"] = (
                f"Image predicts malignant condition "
                f"({confidence*100:.1f}% confidence) but "
                f"clinical symptoms indicate low risk "
                f"(risk score {risk_score}/100, {risk_level}). "
                f"Image prediction may be correct — review."
            )

        elif assessment.get("image_confidence_level") == "LOW":
            conflict["type"]        = "LOW_CONFIDENCE"
            conflict["severity"]    = "LOW"
            conflict["description"] = (
                f"Image confidence is low ({confidence*100:.1f}%). "
                f"Additional clinical correlation required."
            )

        print(f"\n  -- Conflict Detection --")
        print(f"  Conflict type : {conflict['type']}")
        print(f"  Severity      : {conflict['severity']}")
        print(f"  Description   : {conflict['description'][:80]}...")

        return conflict

    # ── Improvement 5 — Confidence-Aware Decision Policy ───────────

    def _make_decision(
        self,
        conflict        : dict,
        confidence      : float,
        clinical_result : dict,
        is_ood          : bool,
    ) -> dict:
        """
        Confidence-aware decision policy.
        Replaces basic PASS/FAIL with clinical decision.

        Decision types:
        CONCORDANT  : Image and symptoms agree, high confidence
        ESCALATE    : Conflict IMAGE_LOW_SYMPTOM_HIGH — urgent
        REVIEW      : Conflict IMAGE_HIGH_SYMPTOM_LOW — review
        UNCERTAIN   : Low confidence, more evidence needed
        OOD_WARNING : Out-of-distribution image detected
        REFER       : High clinical risk, refer to specialist
        ROUTINE     : Low risk, routine monitoring
        """
        risk_level = (
            clinical_result.get("risk_level")
            if clinical_result and
            clinical_result.get("status") == "PASS"
            else None
        )

        if is_ood:
            return {
                "decision"       : "OOD_WARNING",
                "urgency"        : "REFER_SPECIALIST",
                "explanation"    : (
                    "The submitted image shows a condition that "
                    "may be outside the system's 9 trained skin "
                    "disease classes. The AI confidence is below "
                    "the reliable threshold. Please consult a "
                    "dermatologist directly for accurate diagnosis."
                ),
            }

        conflict_type = conflict.get("type", "CONCORDANT")

        if conflict_type == "IMAGE_LOW_SYMPTOM_HIGH":
            return {
                "decision"    : "ESCALATE",
                "urgency"     : "URGENT",
                "explanation" : (
                    "CRITICAL DISCORDANCE DETECTED. The image "
                    "prediction suggests a benign condition but "
                    "patient symptoms strongly indicate malignancy "
                    "risk. Clinical symptoms override the image "
                    "prediction. Urgent specialist referral required."
                ),
            }

        elif conflict_type == "IMAGE_HIGH_SYMPTOM_LOW":
            return {
                "decision"    : "REVIEW",
                "urgency"     : "WITHIN_1_WEEK",
                "explanation" : (
                    "MODERATE DISCORDANCE. Image predicts a "
                    "malignant condition but patient symptoms "
                    "indicate low clinical risk. The image "
                    "prediction may be correct. Clinical review "
                    "and dermoscopic examination within 1 week."
                ),
            }

        elif conflict_type == "LOW_CONFIDENCE":
            return {
                "decision"    : "UNCERTAIN",
                "urgency"     : "WITHIN_4_WEEKS",
                "explanation" : (
                    "Image classification confidence is low. "
                    "Result should be interpreted with caution. "
                    "Clinical correlation and dermoscopic "
                    "examination recommended."
                ),
            }

        # Concordant cases — use risk level for urgency
        if risk_level == "CRITICAL":
            return {
                "decision"    : "REFER",
                "urgency"     : "URGENT",
                "explanation" : (
                    "Image and clinical findings are concordant. "
                    "Clinical risk is CRITICAL. "
                    "Urgent specialist referral required."
                ),
            }
        elif risk_level == "HIGH":
            return {
                "decision"    : "REFER",
                "urgency"     : "WITHIN_1_WEEK",
                "explanation" : (
                    "Image and clinical findings are concordant. "
                    "Clinical risk is HIGH. "
                    "Specialist referral within 1 week."
                ),
            }
        elif risk_level == "MODERATE":
            return {
                "decision"    : "CONCORDANT",
                "urgency"     : "WITHIN_4_WEEKS",
                "explanation" : (
                    "Image and clinical findings are concordant. "
                    "Clinical risk is MODERATE. "
                    "Clinical assessment within 4 weeks."
                ),
            }
        else:
            return {
                "decision"    : "CONCORDANT",
                "urgency"     : "ROUTINE",
                "explanation" : (
                    "Image and clinical findings are concordant. "
                    "Low clinical risk. "
                    "Routine monitoring recommended."
                ),
            }

    # ── Main Entry Point ───────────────────────────────────────────

    def process(
        self,
        image_path : str,
        symptoms   : dict = None
    ) -> dict:
        """
        Main entry point. Patient submits image + symptoms.
        Orchestrator manages the full adaptive pipeline.
        """
        start_time = time.time()
        case_id    = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("\n" + "═" * 60)
        print(f"  MAAI-SkinDx | New Case: {case_id}")
        print(f"  Image  : {Path(image_path).name}")
        print(f"  Symptoms provided: {'Yes' if symptoms else 'No'}")
        print("═" * 60)

        result = {
            "case_id"          : case_id,
            "image_path"       : image_path,
            "symptoms"         : symptoms or {},
            "pipeline_status"  : "RUNNING",
            "agents"           : {},
            "assessment"       : {},
            "conflict"         : {},
            "decision"         : {},
            "final_status"     : None,
            "message"          : None,
            "elapsed_seconds"  : None,
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

        predicted_name  = agent3_result.get(
            "predicted_name",  "Unknown")
        predicted_class = agent3_result.get(
            "predicted_class", "Unknown")
        predicted_index = agent3_result.get("predicted_index", 0)
        confidence      = agent3_result.get("confidence", 0)
        confidence_pct  = confidence * 100

        print(f"\n  Agent 3 → PASS.")
        print(f"  Predicted disease : {predicted_name}")
        print(f"  Confidence        : {confidence_pct:.1f}%")

        # ══════════════════════════════════════════════════════════
        # IMPROVEMENT 1 — OOD Detection
        # Check confidence before continuing pipeline.
        # If below threshold — image likely out of distribution.
        # ══════════════════════════════════════════════════════════
        print(f"\n── OOD Check ──")
        is_ood = self._check_ood(confidence)

        if is_ood:
            print(f"  WARNING: OOD detected — "
                  f"confidence {confidence_pct:.1f}% "
                  f"below threshold "
                  f"{OOD_CONFIDENCE_THRESHOLD*100:.0f}%")
            print(f"  Image may show condition outside "
                  f"9 trained classes (e.g. chickenpox, "
                  f"monkeypox)")
        else:
            print(f"  OOD check PASSED — "
                  f"confidence {confidence_pct:.1f}% "
                  f"above threshold")

        # ══════════════════════════════════════════════════════════
        # IMPROVEMENT 3 — Pre-Fusion Reasoning
        # Assess each agent before combining evidence.
        # ══════════════════════════════════════════════════════════
        print(f"\n── Pre-Fusion Reasoning ──")
        assessment = self._pre_fusion_reasoning(
            agent3_result   = agent3_result,
            confidence      = confidence,
            predicted_index = predicted_index,
            symptoms        = symptoms,
        )
        result["assessment"] = assessment

        # ══════════════════════════════════════════════════════════
        # STEPS 4+5 — Agent 4 + Clinical Agent [parallel]
        # ══════════════════════════════════════════════════════════
        print("\n── Steps 4+5: Agent 4 (XAI) + "
              "Clinical Agent [parallel] ──")

        agent4_result   = None
        clinical_result = None

        def run_agent4():
            return self.agent4.run(
                image_path      = image_path,
                agent3_model    = self.agent3.model,
                agent2_features = feature_vector,
                predicted_index = predicted_index,
                predicted_name  = predicted_class,
                confidence      = confidence,
                use_agent2      = self.agent3.use_agent2,
                case_id         = case_id,
                top_5           = agent3_result.get(
                    "top_5_predictions", []),
            )

        def run_clinical():
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
                        agent4_result = {
                            "status": "FAIL", "reason": str(e)}
                    else:
                        clinical_result = {
                            "status": "FAIL", "reason": str(e)}

        result["agents"]["agent4"]   = agent4_result
        result["agents"]["clinical"] = clinical_result

        # ══════════════════════════════════════════════════════════
        # IMPROVEMENT 4 — Conflict Detection
        # Formally detect disagreement between agents.
        # ══════════════════════════════════════════════════════════
        print(f"\n── Conflict Detection ──")
        conflict = self._detect_conflict(
            predicted_index = predicted_index,
            confidence      = confidence,
            clinical_result = clinical_result,
            assessment      = assessment,
        )
        result["conflict"] = conflict

        # ══════════════════════════════════════════════════════════
        # IMPROVEMENT 5 — Confidence-Aware Decision
        # ══════════════════════════════════════════════════════════
        print(f"\n── Decision Making ──")
        decision = self._make_decision(
            conflict        = conflict,
            confidence      = confidence,
            clinical_result = clinical_result,
            is_ood          = is_ood,
        )
        result["decision"] = decision
        print(f"  Decision  : {decision['decision']}")
        print(f"  Urgency   : {decision['urgency']}")

        # ══════════════════════════════════════════════════════════
        # STEP 6 — Agent 5: RAG Knowledge Retrieval
        # ══════════════════════════════════════════════════════════
        print("\n── Step 6: Agent 5 — RAG Knowledge Retrieval ──")

        focus_type      = None
        xai_explanation = None
        risk_level      = None
        risk_score      = None
        ml_probability  = None

        if agent4_result and agent4_result.get("status") == "PASS":
            focus_stats     = agent4_result.get("focus_stats") or {}
            focus_type      = focus_stats.get("focus_type")
            xai_explanation = agent4_result.get("explanation")

        if clinical_result and clinical_result.get(
                "status") == "PASS":
            risk_level     = clinical_result.get("risk_level")
            risk_score     = clinical_result.get("risk_score")
            ml_probability = clinical_result.get("ml_probability")

        agent5_result = self.agent5.run(
            case_id         = case_id,
            predicted_index = predicted_index,
            predicted_name  = predicted_name,
            confidence      = confidence,
            top_5           = agent3_result.get(
                "top_5_predictions", []),
            risk_level      = risk_level,
            risk_score      = risk_score,
            ml_probability  = ml_probability,
            focus_type      = focus_type,
            xai_explanation = xai_explanation,
        )

        result["agents"]["agent5"] = agent5_result
        print(f"\n  AGENT5 completed — "
              f"status: {agent5_result.get('status', 'UNKNOWN')}")

        if agent5_result.get("status") == "PASS":
            print(f"  Chunks retrieved: "
                  f"{agent5_result.get('chunks_retrieved', 0)}")

        # ── Build final result ──────────────────────────────────────
        elapsed     = round(time.time() - start_time, 2)
        xai_status  = (agent4_result.get("status", "FAIL")
                       if agent4_result else "FAIL")
        overlay     = (agent4_result.get("overlay_path", "")
                       if agent4_result else "")
        explanation = (agent4_result.get("explanation", "")
                       if agent4_result else "")
        clin_status = (clinical_result.get("status", "FAIL")
                       if clinical_result else "FAIL")
        rag_status  = (agent5_result.get("status", "FAIL")
                       if agent5_result else "FAIL")

        clin_msg = (f" Clinical risk: {risk_level}."
                    if risk_level else "")
        rag_msg  = (f" RAG knowledge retrieved "
                    f"({agent5_result.get('chunks_retrieved',0)}"
                    f" chunks)."
                    if rag_status == "PASS" else "")
        ood_msg  = (" ⚠ OOD WARNING: image may be outside "
                    "trained classes." if is_ood else "")

        result.update({
            "pipeline_status"  : "COMPLETE",
            "final_status"     : "PASS",
            "predicted_disease": predicted_name,
            "confidence"       : f"{confidence_pct:.1f}%",
            "is_ood"           : is_ood,
            "top_5_predictions": agent3_result.get(
                "top_5_predictions", []),
            "xai_overlay"      : overlay,
            "xai_status"       : xai_status,
            "xai_explanation"  : explanation,
            "clinical_status"  : clin_status,
            "clinical_risk"    : risk_level,
            "rag_status"       : rag_status,
            "rag_summary"      : agent5_result.get("summary"),
            "message"          : (
                f"Image quality check passed. "
                f"Features extracted ({feature_dim} dims). "
                f"Disease classified: {predicted_name} "
                f"({confidence_pct:.1f}% confidence)."
                f"{ood_msg}"
                f" XAI explanation generated.{clin_msg}"
                f"{rag_msg}"
                f" Decision: {decision['decision']}."
            ),
            "elapsed_seconds"  : elapsed,
        })

        self._print_summary(result)
        return result

    # ── Summary Printer ────────────────────────────────────────────

    def _print_summary(self, result: dict):
        """Print complete pipeline summary with all improvements."""
        print("\n" + "═" * 60)
        print(f"  Pipeline Summary — Case: {result['case_id']}")
        print("═" * 60)
        print(f"  Status  : {result['pipeline_status']}")
        print(f"  Message : {result['message']}")
        print(f"  Time    : {result['elapsed_seconds']}s")

        print(f"\n  Agent Status:")
        for agent_name, agent_result in result["agents"].items():
            status = (agent_result.get("status", "UNKNOWN")
                      if agent_result else "UNKNOWN")
            trust  = AGENT_TRUST_SCORES.get(agent_name, 0.0)
            print(f"  {agent_name.upper():<12}: "
                  f"{status:<8} "
                  f"(trust: {trust:.3f})")

        # OOD warning
        if result.get("is_ood"):
            print(f"\n  ⚠ OOD WARNING ⚠")
            print(f"  Image confidence below {OOD_CONFIDENCE_THRESHOLD*100:.0f}%")
            print(f"  Possible out-of-distribution condition")
            print(f"  e.g. chickenpox, monkeypox, or unknown")

        # Top 5 predictions
        if result.get("top_5_predictions"):
            print(f"\n  Top 5 Disease Predictions:")
            for pred in result["top_5_predictions"]:
                bar = '█' * int(pred['confidence'] * 20)
                print(f"    {pred['rank']}. "
                      f"{pred['class_name']:<35} "
                      f"{pred['confidence_pct']:>6}  {bar}")

        # Assessment
        assess = result.get("assessment", {})
        if assess:
            print(f"\n  Pre-Fusion Assessment:")
            print(f"  Image confidence level : "
                  f"{assess.get('image_confidence_level', '-')}")
            print(f"  Malignant predicted    : "
                  f"{assess.get('image_says_malignant', '-')}")
            print(f"  Clinical data present  : "
                  f"{assess.get('clinical_data_available', '-')}")
            print(f"  Prediction entropy     : "
                  f"{assess.get('prediction_entropy', 0):.4f} "
                  f"({'HIGH' if assess.get('high_entropy') else 'NORMAL'})")

        # Conflict
        conflict = result.get("conflict", {})
        if conflict:
            print(f"\n  Conflict Analysis:")
            print(f"  Type     : {conflict.get('type', '-')}")
            print(f"  Severity : {conflict.get('severity', '-')}")

        # Decision
        decision = result.get("decision", {})
        if decision:
            print(f"\n  Final Decision:")
            print(f"  Decision : {decision.get('decision', '-')}")
            print(f"  Urgency  : {decision.get('urgency', '-')}")
            exp = decision.get("explanation", "")
            if exp:
                print(f"  Rationale: {exp[:100]}...")

        # XAI
        if result.get("xai_overlay"):
            print(f"\n  XAI Overlay : {result['xai_overlay']}")
            print(f"  XAI Status  : {result.get('xai_status','')}")

        # Clinical
        clin = result.get("agents", {}).get("clinical") or {}
        if clin.get("risk_level"):
            print(f"\n  Clinical Risk  : {clin['risk_level']} "
                  f"(score {clin.get('risk_score', 0)}/100)")
            if clin.get("ml_probability") is not None:
                print(f"  ML Probability : "
                      f"{clin['ml_probability']*100:.1f}%")
            print(f"  Recommendation : "
                  f"{clin.get('recommendation', '')}")

        # RAG
        rag = result.get("agents", {}).get("agent5") or {}
        if rag.get("status") == "PASS":
            print(f"\n  RAG Status      : PASS")
            print(f"  Chunks retrieved: "
                  f"{rag.get('chunks_retrieved', 0)}")
            if rag.get("summary"):
                print(f"\n  Clinical Knowledge Summary:")
                for line in rag.get("summary", "").split(". "):
                    if line.strip():
                        print(f"    {line.strip()}.")

        print("═" * 60 + "\n")