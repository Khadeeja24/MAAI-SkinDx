# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Orchestrator — Adaptive Coordination Intelligence
# ══════════════════════════════════════════════════════════════════
# Central coordination agent for the MAAI-SkinDx pipeline.
# Implements the Adaptive Coordination System
# (Phases 1 through 10) and aligns with the Strong
# Coordination Agent Evaluation Guide.
#
# No agent communicates directly with another agent.
# All inter-agent communication passes through this Orchestrator.
#
# Coordination phases implemented:
#
#   Phase 1  — Unified message format
#   Phase 2  — Agent assessment (base trust scores)
#   Phase 4  — Adaptive trust per case
#   Phase 5  — Adaptive fusion (corrected malignancy signal)
#   Phase 6  — Pre-fusion and post-fusion reasoning
#   Phase 7  — Conflict resolution
#   Phase 8  — Decision intelligence (9 decision types)
#   Phase 9  — Action intelligence (Agent 7 clinical report)
#   Phase 10 — Feedback intelligence (Agent 6 background learning)
#
# Pipeline flow:
#   Agent 1  — Image quality check
#   Agent 2  — Visual feature extraction (6916-dim)
#   Agent 3  — Disease classification (9 classes)
#   Pox MLP  — Runs on EVERY case (softmax overconfidence fix)
#              Four outcomes:
#                Agent 3 confident + pox not confident  : normal
#                Agent 3 confident + pox confident      : pox overrides
#                Agent 3 not confident + pox confident  : pox confirmed
#                Agent 3 not confident + pox not confident: true OOD
#   Phase 1  — Unified message wrapping
#   Phase 2+4 — Agent assessment and adaptive trust
#   Agent 4 + Clinical Agent in parallel
#   Phase 5  — Adaptive fusion
#   Phase 6  — Pre-fusion reasoning
#   Phase 7  — Conflict resolution
#   Phase 6  — Post-fusion reasoning
#   Phase 8  — Decision intelligence
#   Agent 5  — RAG knowledge retrieval
#   Agent 7  — Expert clinical report generation
#   Agent 6  — Store case in background (Phase 10)
# ══════════════════════════════════════════════════════════════════

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import time
import math
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
from agents.pox_classifier            import PoxClassifierAgent
from agents.agent6_background         import BackgroundLearningAgent

# ── Phase 2 — Base trust scores ────────────────────────────────────
BASE_TRUST_SCORES = {
    "agent1"  : 0.979,
    "agent2"  : 0.950,
    "agent3"  : 0.742,
    "agent4"  : 0.850,
    "clinical": 0.927,
    "agent5"  : 0.800,
}

OOD_THRESHOLD    = 0.40
POX_THRESHOLD    = 0.65
MALIGNANT_CLASSES = {0, 2, 3}
POX_INDEX_OFFSET  = 100

os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
TRUST_LOG_PATH = os.path.join(
    PROJECT_ROOT, "logs", "trust_evolution.json")


class Orchestrator:

    def __init__(self):
        print("\n" + "=" * 60)
        print("  MAAI-SkinDx | Orchestrator")
        print("  Adaptive Coordination Intelligence")
        print("  Adaptive Coordination System | Phases 1 to 10")
        print("=" * 60)

        print("\n  Loading agents...")
        self.agent1    = ImageQualityAgent()
        self.agent2    = FeatureExtractionAgent()
        self.agent3    = DiagnosisAgent()
        self.agent4    = XAIAgent()
        self.clinical  = ClinicalAgent()
        self.agent5    = RAGAgent()
        self.pox_agent = PoxClassifierAgent()

        # Phase 10 — Agent 6 Background Learning
        self.agent6 = BackgroundLearningAgent()

        # Agent 7 Expert
        self.agent7 = None
        try:
            from agents.agent7_Expert import ClinicalExpertAgent
            self.agent7 = ClinicalExpertAgent()
            print("  Agent 7 Expert loaded.")
        except Exception as e:
            print(f"  Agent 7 Expert not available: {e}")

        # Load base trust scores
        self.trust_scores  = BASE_TRUST_SCORES.copy()
        self.trust_history = []

        # Phase 10 — Resume trust scores from Agent 6 if available
        # This ensures trust scores persist across Orchestrator restarts
        saved_scores = self.agent6.get_trust_scores()
        if saved_scores:
            self.trust_scores.update(saved_scores)
            print(f"  Agent 6 trust scores resumed: {saved_scores}")

        print("\n  Trust scores (active):")
        for name, score in self.trust_scores.items():
            print(f"    {name:<12}: {score:.3f}")
        print("=" * 60 + "\n")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1 — Unified Message Format
    # ══════════════════════════════════════════════════════════════

    def _wrap(
        self,
        agent_name : str,
        raw_result : dict,
        start_time : float,
        confidence : float = None,
    ) -> dict:
        latency = round(time.time() - start_time, 3)
        status  = raw_result.get("status", "UNKNOWN")
        conf    = confidence if confidence is not None else (
            raw_result.get("confidence", 0.0) or 0.0)

        if status == "PASS":
            quality = 0.90
        elif status == "SKIP":
            quality = 0.50
        else:
            quality = 0.10

        return {
            "agent_name"    : agent_name,
            "status"        : status,
            "confidence"    : round(conf, 4),
            "uncertainty"   : round(1.0 - conf, 4),
            "quality_score" : quality,
            "latency_s"     : latency,
            "data_available": status in ("PASS", "SKIP"),
            "raw"           : raw_result,
        }

    # ══════════════════════════════════════════════════════════════
    # PHASE 2 + 4 — Agent Assessment and Adaptive Trust
    # ══════════════════════════════════════════════════════════════

    def _assess_and_adapt(
        self,
        unified_outputs : dict,
        symptoms        : dict,
    ) -> tuple:
        assessment   = {}
        trust_scores = {}

        a3_conf      = unified_outputs.get(
            "agent3", {}).get("confidence", 0.5)
        has_symptoms = (
            symptoms is not None and len(symptoms) > 0)

        for name, wo in unified_outputs.items():
            base_trust  = self.trust_scores.get(name, 0.5)
            quality     = wo.get("quality_score", 0.5)
            uncertainty = wo.get("uncertainty", 0.5)

            if name == "clinical":
                relevance = 1.0 if has_symptoms else 0.3
            elif name == "agent4":
                relevance = min(1.0, a3_conf + 0.2)
            else:
                relevance = 1.0

            t_adaptive = (
                base_trust * quality *
                (1.0 - uncertainty) * relevance)
            t_adaptive = round(min(1.0, max(0.0, t_adaptive)), 4)

            assessment[name] = {
                "base_trust"    : base_trust,
                "quality"       : quality,
                "uncertainty"   : uncertainty,
                "relevance"     : round(relevance, 3),
                "adaptive_trust": t_adaptive,
            }
            trust_scores[name] = t_adaptive

        self.trust_history.append({
            "timestamp"   : datetime.now().isoformat(),
            "trust_scores": trust_scores.copy(),
        })

        return assessment, trust_scores

    # ══════════════════════════════════════════════════════════════
    # PHASE 5 — Adaptive Fusion
    # ══════════════════════════════════════════════════════════════

    def _adaptive_fusion(
        self,
        trust_scores          : dict,
        a3_confidence         : float,
        ml_probability        : float,
        predicted_is_malignant: bool,
    ) -> dict:
        total = sum(trust_scores.values())
        if total <= 0:
            total = 1.0

        weights = {
            name: round(score / total, 4)
            for name, score in trust_scores.items()
        }

        w3    = weights.get("agent3", 0.0)
        wc    = weights.get("clinical", 0.0)
        w_sum = w3 + wc
        if w_sum <= 0:
            w_sum = 1.0

        w3_norm = w3 / w_sum
        wc_norm = wc / w_sum

        if predicted_is_malignant:
            image_mal_signal = a3_confidence
        else:
            image_mal_signal = 1.0 - a3_confidence

        ml_signal = (
            ml_probability
            if ml_probability is not None and ml_probability >= 0
            else 0.0)

        fused_malignancy = round(
            (image_mal_signal * w3_norm) +
            (ml_signal * wc_norm), 4)

        return {
            "weights"             : weights,
            "fused_malignancy"    : fused_malignancy,
            "image_mal_signal"    : round(image_mal_signal, 4),
            "image_weight_norm"   : round(w3_norm, 4),
            "clinical_weight_norm": round(wc_norm, 4),
        }

    # ══════════════════════════════════════════════════════════════
    # PHASE 6 — Pre-Fusion Reasoning
    # ══════════════════════════════════════════════════════════════

    def _pre_fusion_reasoning(
        self,
        agent3_result   : dict,
        confidence      : float,
        predicted_index : int,
        symptoms        : dict,
        assessment      : dict,
    ) -> dict:
        pf = {}

        if confidence >= 0.70:
            pf["image_confidence_level"] = "HIGH"
        elif confidence >= 0.50:
            pf["image_confidence_level"] = "MODERATE"
        else:
            pf["image_confidence_level"] = "LOW"

        pf["image_says_malignant"] = (
            predicted_index in MALIGNANT_CLASSES)
        pf["clinical_data_available"] = (
            symptoms is not None and len(symptoms) > 0)

        top_5 = agent3_result.get("top_5_predictions", [])
        if top_5:
            probs   = [p.get("confidence", 0) for p in top_5]
            entropy = -sum(
                p * math.log(p + 1e-10)
                for p in probs if p > 0)
            pf["prediction_entropy"] = round(entropy, 4)
            pf["high_entropy"]       = entropy > 1.2
        else:
            pf["prediction_entropy"] = 0.0
            pf["high_entropy"]       = False

        a3_assess = assessment.get("agent3", {})
        pf["weighted_confidence"] = round(
            confidence * a3_assess.get("adaptive_trust", 0.742), 4)

        if (pf["image_confidence_level"] == "HIGH"
                and not pf["high_entropy"]):
            pf["image_evidence_strength"] = "STRONG"
        elif pf["image_confidence_level"] == "LOW":
            pf["image_evidence_strength"] = "WEAK"
        else:
            pf["image_evidence_strength"] = "MODERATE"

        if pf["clinical_data_available"]:
            cl_trust = assessment.get(
                "clinical", {}).get("adaptive_trust", 0.5)
            if cl_trust > 0.8:
                pf["clinical_evidence_strength"] = "STRONG"
            elif cl_trust > 0.5:
                pf["clinical_evidence_strength"] = "MODERATE"
            else:
                pf["clinical_evidence_strength"] = "WEAK"
        else:
            pf["clinical_evidence_strength"] = "NOT AVAILABLE"

        print(f"\n  [Pre-Fusion Reasoning]")
        print(f"  Image confidence    : "
              f"{pf['image_confidence_level']} ({confidence*100:.1f}%)")
        print(f"  Image evidence      : {pf['image_evidence_strength']}")
        print(f"  Clinical evidence   : {pf['clinical_evidence_strength']}")
        print(f"  Malignant predicted : {pf['image_says_malignant']}")
        print(f"  Prediction entropy  : {pf['prediction_entropy']:.4f} "
              f"({'HIGH' if pf['high_entropy'] else 'NORMAL'})")
        print(f"  Weighted confidence : {pf['weighted_confidence']:.4f}")

        return pf

    # ══════════════════════════════════════════════════════════════
    # PHASE 7 — Conflict Resolution
    # ══════════════════════════════════════════════════════════════

    def _conflict_resolution(
        self,
        predicted_index : int,
        confidence      : float,
        clinical_result : dict,
        prefusion       : dict,
        fusion          : dict,
    ) -> dict:
        conflict = {
            "type"        : "CONCORDANT",
            "severity"    : "NONE",
            "description" : (
                "Image and clinical evidence are in agreement. "
                "No discordance detected."),
            "resolved_by" : "CONCORDANT",
            "vote_entropy": 0.0,
        }

        if not clinical_result or clinical_result.get(
                "status") == "SKIP":
            conflict["type"]        = "NO_CLINICAL_DATA"
            conflict["description"] = (
                "No symptom data was submitted. "
                "The diagnostic decision is based on image evidence only.")
            return conflict

        ml_prob    = clinical_result.get("ml_probability")
        risk_level = clinical_result.get("risk_level", "LOW")
        risk_score = clinical_result.get("risk_score", 0)

        image_malignant   = predicted_index in MALIGNANT_CLASSES
        symptom_malignant = (
            (ml_prob is not None and ml_prob > 0.60) or
            risk_level in ("HIGH", "CRITICAL"))
        fused_mal = fusion.get("fused_malignancy", 0.0)

        p_image    = confidence if image_malignant else 1.0 - confidence
        p_clinical = ml_prob if ml_prob is not None else 0.5
        p_avg      = (p_image + p_clinical) / 2.0
        if 0 < p_avg < 1:
            vote_entropy = round(
                -p_avg * math.log(p_avg + 1e-10)
                - (1 - p_avg) * math.log(1 - p_avg + 1e-10), 4)
        else:
            vote_entropy = 0.0
        conflict["vote_entropy"] = vote_entropy

        if not image_malignant and symptom_malignant:
            ml_pct = (f"{ml_prob*100:.1f}%"
                      if ml_prob is not None else "N/A")
            conflict.update({
                "type"       : "IMAGE_LOW_SYMPTOM_HIGH",
                "severity"   : "HIGH",
                "description": (
                    f"Image predicts a benign condition at "
                    f"{confidence*100:.1f}% confidence but "
                    f"clinical symptoms indicate elevated "
                    f"malignancy risk (ML probability {ml_pct}, "
                    f"rule-based score {risk_score}/100, "
                    f"risk level {risk_level}). "
                    f"Clinical evidence overrides image prediction."),
                "resolved_by": "CLINICAL_OVERRIDE",
            })
        elif image_malignant and not symptom_malignant:
            conflict.update({
                "type"       : "IMAGE_HIGH_SYMPTOM_LOW",
                "severity"   : "MODERATE",
                "description": (
                    f"Image predicts a malignant condition at "
                    f"{confidence*100:.1f}% confidence but "
                    f"clinical symptoms indicate low risk "
                    f"(rule-based score {risk_score}/100, "
                    f"risk level {risk_level}). "
                    f"Image prediction requires clinical review."),
                "resolved_by": "REVIEW_REQUIRED",
            })
        elif fused_mal > 0.60 and not image_malignant:
            conflict.update({
                "type"       : "FUSED_SIGNAL_HIGH",
                "severity"   : "MODERATE",
                "description": (
                    f"The fused malignancy signal ({fused_mal:.3f}) "
                    f"exceeds the threshold despite a benign image "
                    f"prediction. Combined weighted evidence "
                    f"warrants clinical review."),
                "resolved_by": "FUSION_ESCALATION",
            })
        elif prefusion.get("image_confidence_level") == "LOW":
            conflict.update({
                "type"       : "LOW_CONFIDENCE",
                "severity"   : "LOW",
                "description": (
                    f"Image classification confidence is low "
                    f"({confidence*100:.1f}%). "
                    f"Image evidence alone is insufficient "
                    f"for a reliable diagnostic decision."),
                "resolved_by": "INSUFFICIENT_EVIDENCE",
            })

        print(f"\n  [Conflict Resolution]")
        print(f"  Type        : {conflict['type']}")
        print(f"  Severity    : {conflict['severity']}")
        print(f"  Vote entropy: {conflict['vote_entropy']:.4f}")
        print(f"  Resolved by : {conflict['resolved_by']}")

        return conflict

    # ══════════════════════════════════════════════════════════════
    # PHASE 6 — Post-Fusion Reasoning
    # ══════════════════════════════════════════════════════════════

    def _post_fusion_reasoning(
        self,
        conflict  : dict,
        fusion    : dict,
        prefusion : dict,
        is_ood    : bool,
    ) -> dict:
        pfr           = {}
        conflict_type = conflict.get("type", "CONCORDANT")
        fused_mal     = fusion.get("fused_malignancy", 0.0)
        img_str       = prefusion.get(
            "image_evidence_strength", "MODERATE")

        if is_ood:
            pfr["evidence_sufficient"] = False
            pfr["evidence_quality"]    = "INSUFFICIENT"
            pfr["evidence_assessment"] = (
                "Image confidence is below the OOD threshold. "
                "A reliable diagnostic decision cannot be made.")
        elif conflict_type == "CONCORDANT" and img_str == "STRONG":
            pfr["evidence_sufficient"] = True
            pfr["evidence_quality"]    = "STRONG"
            pfr["evidence_assessment"] = (
                "Image and clinical evidence agree strongly. "
                "Combined evidence is sufficient.")
        elif conflict_type in (
                "IMAGE_LOW_SYMPTOM_HIGH", "FUSED_SIGNAL_HIGH"):
            pfr["evidence_sufficient"] = True
            pfr["evidence_quality"]    = "CONFLICTED"
            pfr["evidence_assessment"] = (
                "Clinical evidence overrides image prediction. "
                "Conflict detected and resolved. Escalation is indicated.")
        elif conflict_type == "LOW_CONFIDENCE":
            pfr["evidence_sufficient"] = False
            pfr["evidence_quality"]    = "WEAK"
            pfr["evidence_assessment"] = (
                "Image evidence is insufficient. "
                "Clinical assessment or dermoscopy is required.")
        else:
            pfr["evidence_sufficient"] = True
            pfr["evidence_quality"]    = "ADEQUATE"
            pfr["evidence_assessment"] = (
                "Combined evidence is adequate to support "
                "a provisional diagnostic decision.")

        if fused_mal > 0.75:
            pfr["fused_signal_level"] = "HIGH"
        elif fused_mal > 0.50:
            pfr["fused_signal_level"] = "MODERATE"
        elif fused_mal > 0.25:
            pfr["fused_signal_level"] = "LOW"
        else:
            pfr["fused_signal_level"] = "BENIGN"

        print(f"\n  [Post-Fusion Reasoning]")
        print(f"  Evidence quality    : {pfr['evidence_quality']}")
        print(f"  Evidence sufficient : {pfr['evidence_sufficient']}")
        print(f"  Fused signal level  : "
              f"{pfr['fused_signal_level']} ({fused_mal:.4f})")

        return pfr

    # ══════════════════════════════════════════════════════════════
    # PHASE 8 — Decision Intelligence
    # ══════════════════════════════════════════════════════════════

    def _decision_intelligence(
        self,
        conflict        : dict,
        prefusion       : dict,
        postfusion      : dict,
        fusion          : dict,
        clinical_result : dict,
        is_ood          : bool,
        pox_result      : dict = None,
    ) -> dict:
        conflict_type = conflict.get("type", "CONCORDANT")
        risk_level    = (
            clinical_result.get("risk_level")
            if clinical_result and
            clinical_result.get("status") == "PASS"
            else None)
        evidence_ok = postfusion.get("evidence_sufficient", True)
        fused_mal   = fusion.get("fused_malignancy", 0.0)

        if is_ood and (
            pox_result is None or
            not pox_result.get("pox_detected")
        ):
            return {
                "decision"  : "OOD_WARNING",
                "urgency"   : "REFER_SPECIALIST",
                "confidence": "LOW",
                "rationale" : (
                    "The submitted image has a top-1 confidence "
                    "below the reliable threshold of 0.40 and the "
                    "pox classifier did not identify a known pox "
                    "family disease with sufficient confidence. "
                    "Direct specialist consultation is required."),
            }

        if conflict_type == "IMAGE_LOW_SYMPTOM_HIGH":
            return {
                "decision"  : "ESCALATE",
                "urgency"   : "URGENT",
                "confidence": "HIGH",
                "rationale" : (
                    "Critical discordance detected. The image "
                    "prediction indicates a benign condition "
                    "but patient symptom evidence strongly "
                    "suggests elevated malignancy risk. "
                    "Clinical symptom evidence overrides the "
                    "image prediction. Urgent specialist referral is required."),
            }

        if conflict_type in ("IMAGE_HIGH_SYMPTOM_LOW", "FUSED_SIGNAL_HIGH"):
            return {
                "decision"  : "REVIEW",
                "urgency"   : "WITHIN_1_WEEK",
                "confidence": "MODERATE",
                "rationale" : (
                    "Moderate discordance detected between "
                    "image and clinical evidence. Clinical "
                    "review and dermoscopic examination "
                    "within one week is recommended."),
            }

        if not evidence_ok and conflict_type == "LOW_CONFIDENCE":
            return {
                "decision"  : "REQUEST_EVIDENCE",
                "urgency"   : "WITHIN_4_WEEKS",
                "confidence": "LOW",
                "rationale" : (
                    "Image classification confidence is "
                    "insufficient to support a reliable "
                    "diagnostic decision. Additional evidence is required."),
            }

        if not evidence_ok:
            return {
                "decision"  : "UNCERTAIN",
                "urgency"   : "WITHIN_4_WEEKS",
                "confidence": "LOW",
                "rationale" : (
                    "Combined evidence quality is insufficient "
                    "for a confident diagnostic decision. "
                    "Clinical assessment is recommended."),
            }

        if risk_level == "CRITICAL":
            return {
                "decision"  : "REFER_URGENT",
                "urgency"   : "URGENT",
                "confidence": "HIGH",
                "rationale" : (
                    "Image and clinical evidence are concordant. "
                    "Clinical risk level is CRITICAL. "
                    "Urgent specialist referral is required."),
            }

        if risk_level == "HIGH":
            return {
                "decision"  : "REFER",
                "urgency"   : "WITHIN_1_WEEK",
                "confidence": "HIGH",
                "rationale" : (
                    "Image and clinical evidence are concordant. "
                    "Clinical risk level is HIGH. "
                    "Specialist referral within one week."),
            }

        if risk_level == "MODERATE" or fused_mal > 0.40:
            return {
                "decision"  : "CONCORDANT",
                "urgency"   : "WITHIN_4_WEEKS",
                "confidence": "MODERATE",
                "rationale" : (
                    "Image and clinical evidence are concordant. "
                    "Clinical risk is MODERATE. "
                    "Clinical assessment within four weeks."),
            }

        return {
            "decision"  : "ROUTINE",
            "urgency"   : "ROUTINE",
            "confidence": "HIGH",
            "rationale" : (
                "Image and clinical evidence are concordant. "
                "Clinical risk is LOW. "
                "Routine monitoring is recommended."),
        }

    # ══════════════════════════════════════════════════════════════
    # PHASE 10 — Feedback Intelligence (Agent 6)
    # ══════════════════════════════════════════════════════════════

    def submit_outcome(
        self,
        case_id         : str,
        confirmed_class : int,
        confirmed_name  : str = "",
        outcome_source  : str = "biopsy",
    ) -> dict:
        """
        Submit a confirmed clinical outcome for a previous case.
        Called by clinician after biopsy or specialist verdict.
        Updates agent trust scores based on confirmed accuracy.

        Args:
            case_id         : Case ID from a previous pipeline run
            confirmed_class : Confirmed diagnosis class index (0-8)
            confirmed_name  : Confirmed diagnosis name
            outcome_source  : biopsy / specialist / follow_up

        Returns:
            Dict of updated trust scores.
        """
        updated = self.agent6.submit_outcome(
            case_id         = case_id,
            confirmed_class = confirmed_class,
            confirmed_name  = confirmed_name,
            outcome_source  = outcome_source,
        )

        if updated:
            self.trust_scores.update(updated)
            print(f"  Orchestrator trust scores updated: {updated}")

        return updated

    # ══════════════════════════════════════════════════════════════
    # MAIN PIPELINE
    # ══════════════════════════════════════════════════════════════

    def process(
        self,
        image_path : str,
        symptoms   : dict = None,
    ) -> dict:
        start_time = time.time()
        case_id    = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("\n" + "=" * 60)
        print(f"  MAAI-SkinDx | Case: {case_id}")
        print(f"  Image    : {Path(image_path).name}")
        print(f"  Symptoms : {'Yes' if symptoms else 'No'}")
        print("=" * 60)

        result = {
            "case_id"         : case_id,
            "image_path"      : image_path,
            "symptoms"        : symptoms or {},
            "pipeline_status" : "RUNNING",
            "agents"          : {},
            "unified_outputs" : {},
            "assessment"      : {},
            "trust_scores"    : {},
            "fusion"          : {},
            "prefusion"       : {},
            "postfusion"      : {},
            "conflict"        : {},
            "decision"        : {},
            "final_status"    : None,
            "message"         : None,
            "elapsed_seconds" : None,
        }

        # ── Agent 1 ────────────────────────────────────────────────
        print("\n-- Step 1: Agent 1 -- Image Quality --")
        t1            = time.time()
        agent1_result = self.agent1.run(image_path)
        a1_conf       = 1.0 if agent1_result["status"] == "PASS" else 0.0
        w1 = self._wrap("agent1", agent1_result, t1, confidence=a1_conf)
        result["agents"]["agent1"]          = agent1_result
        result["unified_outputs"]["agent1"] = w1

        if agent1_result["status"] == "FAIL":
            elapsed = round(time.time() - start_time, 2)
            result.update({
                "pipeline_status": "REJECTED",
                "final_status"   : "FAIL",
                "message"        : agent1_result["reason"],
                "elapsed_seconds": elapsed,
            })
            self.agent6.store_case(result)
            self._print_summary(result)
            return result

        print(f"  Agent 1 -- PASS | conf={a1_conf:.2f} | {w1['latency_s']}s")

        # ── Agent 2 ────────────────────────────────────────────────
        print("\n-- Step 2: Agent 2 -- Feature Extraction --")
        t2            = time.time()
        agent2_result = self.agent2.run(image_path)
        a2_conf       = 1.0 if agent2_result["status"] == "PASS" else 0.0
        w2 = self._wrap("agent2", agent2_result, t2, confidence=a2_conf)
        result["agents"]["agent2"]          = agent2_result
        result["unified_outputs"]["agent2"] = w2

        if agent2_result["status"] == "FAIL":
            elapsed = round(time.time() - start_time, 2)
            result.update({
                "pipeline_status": "ERROR",
                "final_status"   : "FAIL",
                "message"        : "Feature extraction failed.",
                "elapsed_seconds": elapsed,
            })
            self.agent6.store_case(result)
            self._print_summary(result)
            return result

        feature_vector = agent2_result.get("feature_vector")
        feature_dim    = agent2_result.get("feature_dim", 0)
        print(f"  Agent 2 -- PASS | conf={a2_conf:.2f} | "
              f"{feature_dim} dims | {w2['latency_s']}s")

        # ── Agent 3 ────────────────────────────────────────────────
        print("\n-- Step 3: Agent 3 -- Classification --")
        t3            = time.time()
        agent3_result = self.agent3.run(
            image_path      = image_path,
            agent2_features = feature_vector,
        )
        confidence = agent3_result.get("confidence", 0.0)
        w3 = self._wrap("agent3", agent3_result, t3, confidence=confidence)
        result["agents"]["agent3"]          = agent3_result
        result["unified_outputs"]["agent3"] = w3

        if agent3_result["status"] == "FAIL":
            elapsed = round(time.time() - start_time, 2)
            result.update({
                "pipeline_status": "ERROR",
                "final_status"   : "FAIL",
                "message"        : "Classification failed.",
                "elapsed_seconds": elapsed,
            })
            self.agent6.store_case(result)
            self._print_summary(result)
            return result

        predicted_name  = agent3_result.get("predicted_name",  "Unknown")
        predicted_class = agent3_result.get("predicted_class", "Unknown")
        predicted_index = agent3_result.get("predicted_index", 0)
        confidence_pct  = confidence * 100
        is_malignant    = predicted_index in MALIGNANT_CLASSES
        is_ood          = confidence < OOD_THRESHOLD

        print(f"  Agent 3 -- PASS | "
              f"{predicted_name} ({confidence_pct:.1f}%) | "
              f"{'MALIGNANT' if is_malignant else 'benign'} | "
              f"{w3['latency_s']}s")

        # ── Pox Classifier — runs on every case ───────────────────
        print(f"\n-- Pox Classifier (runs on every case) --")
        t_pox     = time.time()
        pox_result = self.pox_agent.run(feature_vector)
        result["agents"]["pox"] = pox_result

        pox_detected = pox_result.get("pox_detected", False)
        pox_name     = pox_result.get("predicted_name", "?")
        pox_conf     = pox_result.get("confidence") or 0.0
        pox_conf_pct = pox_conf * 100

        print(f"  Pox MLP: {pox_name} ({pox_conf_pct:.1f}%) | "
              f"detected={pox_detected} | "
              f"{round(time.time()-t_pox, 3)}s")

        # ── Four-way pox decision ─────────────────────────────────
        print(f"\n-- OOD + Pox Decision --")
        pox_override = False

        if not is_ood and not pox_detected:
            print(f"  Agent 3 confident ({confidence_pct:.1f}%) "
                  f"| Pox below threshold ({pox_conf_pct:.1f}%)")
            print(f"  Decision: Normal pipeline — trust Agent 3")

        elif not is_ood and pox_detected:
            pox_override = True
            print(f"  Agent 3 confident ({confidence_pct:.1f}%) "
                  f"BUT Pox also confident ({pox_conf_pct:.1f}%)")
            print(f"  Overconfident softmax detected — "
                  f"Pox MLP specialist overrides Agent 3")

        elif is_ood and pox_detected:
            pox_override = True
            print(f"  Agent 3 uncertain ({confidence_pct:.1f}%) "
                  f"| Pox confident ({pox_conf_pct:.1f}%)")
            print(f"  POX CONFIRMED — pipeline continues as pox case")

        else:
            print(f"  Agent 3 uncertain ({confidence_pct:.1f}%) "
                  f"| Pox also uncertain ({pox_conf_pct:.1f}%)")
            print(f"  TRUE OOD WARNING — unknown condition")

        if pox_override:
            predicted_name  = pox_result["predicted_name"]
            predicted_index = (
                pox_result["predicted_index"] + POX_INDEX_OFFSET)
            confidence      = pox_conf
            confidence_pct  = pox_conf_pct
            is_malignant    = False
            is_ood          = False
            result["unified_outputs"]["agent3"]["confidence"] = (
                round(confidence, 4))
            result["unified_outputs"]["agent3"]["uncertainty"] = (
                round(1.0 - confidence, 4))
            print(f"  Override applied: {predicted_name} "
                  f"({confidence_pct:.1f}%)")

        # ── Parallel: Agent 4 + Clinical ───────────────────────────
        print("\n-- Step 4+5: Agent 4 + Clinical [parallel] --")

        agent4_result   = None
        clinical_result = None

        def run_agent4():
            # FIX 3: Use pox-corrected values.
            # predicted_index and predicted_name are updated by the
            # pox override block above before this closure executes.
            # Using agent3_result.get() here would bypass the override
            # and explain the wrong disease when pox is detected.
            return self.agent4.run(
                image_path      = image_path,
                agent3_model    = self.agent3.model,
                agent2_features = feature_vector,
                predicted_index = predicted_index,   # post-pox corrected
                predicted_name  = predicted_name,    # post-pox corrected
                confidence      = confidence,        # post-pox corrected
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

        t45 = time.time()
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
                        agent4_result = res
                    else:
                        clinical_result = res
                    print(f"  {name.upper()} -- "
                          f"{res.get('status','UNKNOWN')}")
                except Exception as e:
                    print(f"  {name.upper()} FAILED: {e}")
                    if name == "agent4":
                        agent4_result = {"status": "FAIL", "reason": str(e)}
                    else:
                        clinical_result = {"status": "FAIL", "reason": str(e)}

        result["agents"]["agent4"]   = agent4_result
        result["agents"]["clinical"] = clinical_result

        ml_prob = (
            clinical_result.get("ml_probability")
            if clinical_result and
            clinical_result.get("status") == "PASS"
            else None)

        focus_stats    = (agent4_result or {}).get("focus_stats") or {}
        focus_type_val = focus_stats.get("focus_type", "")
        if (agent4_result or {}).get("status") == "PASS":
            a4_conf = (0.85 if focus_type_val in (
                "focus_on_lesion", "focus_on_border") else 0.60)
        else:
            a4_conf = 0.0

        w4 = self._wrap("agent4", agent4_result or {}, t45, confidence=a4_conf)
        wc = self._wrap(
            "clinical",
            {**(clinical_result or {}), "confidence": ml_prob or 0.0},
            t45, confidence=ml_prob or 0.0)

        result["unified_outputs"]["agent4"]   = w4
        result["unified_outputs"]["clinical"] = wc

        # ── Phase 2+4 ──────────────────────────────────────────────
        print(f"\n-- Phase 2+4: Agent Assessment --")
        assessment, trust_scores = self._assess_and_adapt(
            result["unified_outputs"], symptoms)
        result["assessment"]   = assessment
        result["trust_scores"] = trust_scores

        for name, a in assessment.items():
            print(f"  {name:<12}: "
                  f"base={a['base_trust']:.3f} "
                  f"adaptive={a['adaptive_trust']:.3f} "
                  f"relevance={a['relevance']:.2f}")

        # ── Phase 5 ────────────────────────────────────────────────
        print(f"\n-- Phase 5: Adaptive Fusion --")
        fusion = self._adaptive_fusion(
            trust_scores           = trust_scores,
            a3_confidence          = confidence,
            ml_probability         = ml_prob,
            predicted_is_malignant = is_malignant,
        )
        result["fusion"] = fusion
        print(f"  Fusion weights:")
        for name, w in fusion["weights"].items():
            bar = "=" * int(w * 30)
            print(f"    {name:<12}: {w:.4f}  {bar}")
        print(f"  Image malignancy signal : "
              f"{fusion['image_mal_signal']:.4f} "
              f"({'MALIGNANT class' if is_malignant else 'benign class'})")
        print(f"  Fused malignancy signal : {fusion['fused_malignancy']:.4f}")

        # ── Phase 6 pre ────────────────────────────────────────────
        print(f"\n-- Phase 6: Pre-Fusion Reasoning --")
        prefusion = self._pre_fusion_reasoning(
            agent3_result   = agent3_result,
            confidence      = confidence,
            predicted_index = predicted_index,
            symptoms        = symptoms,
            assessment      = assessment,
        )
        result["prefusion"] = prefusion

        # ── Phase 7 ────────────────────────────────────────────────
        print(f"\n-- Phase 7: Conflict Resolution --")
        conflict = self._conflict_resolution(
            predicted_index = predicted_index,
            confidence      = confidence,
            clinical_result = clinical_result,
            prefusion       = prefusion,
            fusion          = fusion,
        )
        result["conflict"] = conflict

        # ── Phase 6 post ───────────────────────────────────────────
        print(f"\n-- Phase 6: Post-Fusion Reasoning --")
        postfusion = self._post_fusion_reasoning(
            conflict  = conflict,
            fusion    = fusion,
            prefusion = prefusion,
            is_ood    = is_ood,
        )
        result["postfusion"] = postfusion

        # ── Phase 8 ────────────────────────────────────────────────
        print(f"\n-- Phase 8: Decision Intelligence --")
        decision = self._decision_intelligence(
            conflict        = conflict,
            prefusion       = prefusion,
            postfusion      = postfusion,
            fusion          = fusion,
            clinical_result = clinical_result,
            is_ood          = is_ood,
            pox_result      = pox_result,
        )
        result["decision"] = decision
        print(f"  Decision   : {decision['decision']}")
        print(f"  Urgency    : {decision['urgency']}")
        print(f"  Confidence : {decision['confidence']}")

        # ── Extract for downstream ─────────────────────────────────
        focus_type      = focus_stats.get("focus_type")
        xai_explanation = (
            (agent4_result or {}).get("explanation")
            if (agent4_result or {}).get("status") == "PASS"
            else None)
        risk_level   = None
        risk_score   = None
        risk_factors = []

        if clinical_result and clinical_result.get("status") == "PASS":
            risk_level   = clinical_result.get("risk_level")
            risk_score   = clinical_result.get("risk_score")
            risk_factors = clinical_result.get("risk_factors", [])

        # ── Agent 5 ────────────────────────────────────────────────
        print("\n-- Step 6: Agent 5 -- RAG Knowledge --")
        t5 = time.time()
        agent5_result = self.agent5.run(
            case_id         = case_id,
            predicted_index = predicted_index,
            predicted_name  = predicted_name,
            confidence      = confidence,
            top_5           = agent3_result.get("top_5_predictions", []),
            risk_level      = risk_level,
            risk_score      = risk_score,
            ml_probability  = ml_prob,
            focus_type      = focus_type,
            xai_explanation = xai_explanation,
        )

        chunks  = agent5_result.get("chunks_retrieved", 0)
        a5_conf = (min(1.0, chunks / 5.0)
                   if agent5_result.get("status") == "PASS" else 0.0)
        w5 = self._wrap("agent5", agent5_result, t5, confidence=a5_conf)
        result["agents"]["agent5"]          = agent5_result
        result["unified_outputs"]["agent5"] = w5

        print(f"  Agent 5 -- {agent5_result.get('status')} | "
              f"conf={a5_conf:.2f} | {chunks} chunks | {w5['latency_s']}s")

        # ── Phase 9 — Agent 7 ──────────────────────────────────────
        agent7_result = None
        if self.agent7 is not None:
            print("\n-- Step 7: Agent 7 Expert -- Clinical Report --")
            t7 = time.time()
            agent7_result = self.agent7.run(
                case_id            = case_id,
                predicted_name     = predicted_name,
                predicted_index    = predicted_index,
                confidence         = confidence,
                top_5              = agent3_result.get(
                    "top_5_predictions", []),
                risk_level         = risk_level,
                risk_score         = risk_score or 0,
                ml_probability     = ml_prob,
                risk_factors       = risk_factors,
                focus_type         = focus_type,
                xai_explanation    = xai_explanation,
                rag_summary        = agent5_result.get("summary"),
                conflict_type      = conflict.get("type", "CONCORDANT"),
                conflict_severity  = conflict.get("severity", "NONE"),
                conflict_desc      = conflict.get("description", ""),
                decision           = decision.get("decision", "ROUTINE"),
                urgency            = decision.get("urgency", "ROUTINE"),
                decision_rationale = decision.get("rationale", ""),
                symptoms           = symptoms,
                fusion_weights     = fusion.get("weights", {}),
                fused_malignancy   = fusion.get("fused_malignancy", 0.0),
            )

            a7_conf = (1.0 if (agent7_result or {}).get("status") == "PASS"
                       else 0.0)
            w7 = self._wrap("agent7", agent7_result or {}, t7,
                            confidence=a7_conf)
            result["agents"]["agent7"]          = agent7_result
            result["unified_outputs"]["agent7"] = w7
            print(f"  Agent 7 -- "
                  f"{(agent7_result or {}).get('status','UNKNOWN')} "
                  f"| conf={a7_conf:.2f} | {w7['latency_s']}s")

        # ── Build final result ─────────────────────────────────────
        elapsed = round(time.time() - start_time, 2)

        pox_msg = ""
        if pox_override:
            pox_msg = (f" POX DETECTED: {predicted_name} "
                       f"({confidence_pct:.1f}%).")
        ood_msg = ""
        if is_ood:
            ood_msg = " TRUE OOD WARNING: specialist referral."

        result.update({
            "pipeline_status"  : "COMPLETE",
            "final_status"     : "PASS",
            "predicted_disease": predicted_name,
            "confidence"       : f"{confidence_pct:.1f}%",
            "is_ood"           : is_ood,
            "is_malignant"     : is_malignant,
            "pox_result"       : pox_result,
            "pox_override"     : pox_override,
            "top_5_predictions": agent3_result.get("top_5_predictions", []),
            "clinical_risk"    : risk_level,
            "rag_summary"      : agent5_result.get("summary"),
            "elapsed_seconds"  : elapsed,
            "message"          : (
                f"Pipeline complete. "
                f"Diagnosis: {predicted_name} "
                f"({confidence_pct:.1f}% | "
                f"{'MALIGNANT' if is_malignant else 'benign'})."
                f"{pox_msg}{ood_msg} "
                f"Conflict: {conflict.get('type')}. "
                f"Decision: {decision.get('decision')}. "
                f"Urgency: {decision.get('urgency')}. "
                f"Time: {elapsed}s."
            ),
        })

        # ── Phase 10 — Agent 6 Background Learning ─────────────────
        # Store every completed case silently in the background.
        # When confirmed outcomes are submitted via submit_outcome(),
        # Agent 6 updates trust scores automatically.
        self.agent6.store_case(result)

        self._print_summary(result)
        return result

    # ══════════════════════════════════════════════════════════════
    # SUMMARY PRINTER
    # ══════════════════════════════════════════════════════════════

    def _print_summary(self, result: dict):
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  Pipeline Summary -- {result['case_id']}")
        print(sep)
        print(f"  Status : {result['pipeline_status']}")
        print(f"  Time   : {result['elapsed_seconds']}s")
        print(f"  Message: {result['message']}")

        print(f"\n  Agent Status (Phase 1 Unified Format):")
        for name, wo in result.get("unified_outputs", {}).items():
            a     = result.get("assessment", {}).get(name, {})
            trust = (a.get("adaptive_trust")
                    if a else BASE_TRUST_SCORES.get(name, 0.0))
            label = "adaptive" if a else "base"
            print(
                f"  {name.upper():<12}: "
                f"{wo.get('status','?'):<6} | "
                f"conf={wo.get('confidence',0):.3f} | "
                f"unc={wo.get('uncertainty',0):.3f} | "
                f"trust={trust:.3f} ({label}) | "
                f"{wo.get('latency_s',0):.2f}s"
            )

        pox = result.get("pox_result") or {}
        if pox:
            override = result.get("pox_override", False)
            print(f"\n  Pox Classifier "
                  f"({'OVERRIDE APPLIED' if override else 'no override'}):")
            print(f"    Detected  : {pox.get('pox_detected')}")
            print(f"    Disease   : {pox.get('predicted_name','?')}")
            print(f"    Confidence: {(pox.get('confidence') or 0)*100:.1f}%")
            if pox.get("all_probs"):
                for cls, prob in pox["all_probs"].items():
                    bar = "=" * int(prob * 25)
                    print(f"      {cls:<15}: {prob*100:.1f}%  {bar}")

        if result.get("is_ood"):
            print(f"\n  TRUE OOD WARNING: specialist referral")

        if result.get("top_5_predictions"):
            print(f"\n  Top 5 Predictions (Agent 3):")
            for p in result["top_5_predictions"]:
                bar = "=" * int(p["confidence"] * 20)
                print(f"    {p['rank']}. "
                      f"{p['class_name']:<35} "
                      f"{p['confidence_pct']:>6}  {bar}")

        fusion = result.get("fusion", {})
        if fusion.get("weights"):
            print(f"\n  Phase 5 -- Adaptive Fusion:")
            for name, w in fusion["weights"].items():
                bar = "=" * int(w * 30)
                print(f"    {name:<12}: {w:.4f}  {bar}")
            print(f"  Image signal (corrected): "
                  f"{fusion.get('image_mal_signal',0):.4f}")
            print(f"  Fused malignancy signal : "
                  f"{fusion.get('fused_malignancy',0):.4f}")

        pf = result.get("prefusion", {})
        if pf:
            print(f"\n  Phase 6 -- Pre-Fusion Reasoning:")
            print(f"    Image evidence    : "
                  f"{pf.get('image_evidence_strength','-')}")
            print(f"    Clinical evidence : "
                  f"{pf.get('clinical_evidence_strength','-')}")
            print(f"    Entropy           : "
                  f"{pf.get('prediction_entropy',0):.4f} "
                  f"({'HIGH' if pf.get('high_entropy') else 'NORMAL'})")

        pfr = result.get("postfusion", {})
        if pfr:
            print(f"\n  Phase 6 -- Post-Fusion Reasoning:")
            print(f"    Evidence quality  : {pfr.get('evidence_quality','-')}")
            print(f"    Fused signal      : {pfr.get('fused_signal_level','-')}")

        conflict = result.get("conflict", {})
        if conflict:
            print(f"\n  Phase 7 -- Conflict Resolution:")
            print(f"    Type        : {conflict.get('type','-')}")
            print(f"    Severity    : {conflict.get('severity','-')}")
            print(f"    Vote entropy: {conflict.get('vote_entropy',0):.4f}")
            print(f"    Resolved by : {conflict.get('resolved_by','-')}")

        dec = result.get("decision", {})
        if dec:
            print(f"\n  Phase 8 -- Decision Intelligence:")
            print(f"    Decision    : {dec.get('decision','-')}")
            print(f"    Urgency     : {dec.get('urgency','-')}")
            print(f"    Confidence  : {dec.get('confidence','-')}")
            rat = dec.get("rationale", "")
            if rat:
                print(f"    Rationale   : {rat[:90]}...")

        clin = result.get("agents", {}).get("clinical") or {}
        if clin.get("risk_level"):
            print(f"\n  Clinical Assessment:")
            print(f"    Risk level  : {clin['risk_level']} "
                  f"(score {clin.get('risk_score',0)}/100)")
            if clin.get("ml_probability") is not None:
                print(f"    ML prob     : {clin['ml_probability']*100:.1f}%")
            print(f"    Recommend   : {clin.get('recommendation','')}")

        rag = result.get("agents", {}).get("agent5") or {}
        if rag.get("status") == "PASS":
            print(f"\n  Agent 5 -- RAG Knowledge:")
            print(f"    Chunks : {rag.get('chunks_retrieved',0)}")
            if rag.get("summary"):
                print(f"    Summary:")
                for line in rag["summary"].split(". "):
                    if line.strip():
                        print(f"      {line.strip()}.")

        a7 = result.get("agents", {}).get("agent7") or {}
        if a7.get("status") == "PASS":
            print(f"\n  Phase 9 -- Agent 7 Expert Report:")
            print(f"    Status : PASS")
            print(f"    Saved  : {a7.get('report_path','')}")

        # Agent 6 summary
        print(f"\n  Phase 10 -- Agent 6 Background Learning:")
        print(f"    Cases stored : {self.agent6.get_case_count()}")
        print(f"    Outcomes     : {self.agent6.get_outcome_count()}")

        print(f"{sep}\n")
