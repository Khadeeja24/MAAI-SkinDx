# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Orchestrator — Adaptive Coordination Intelligence
# ══════════════════════════════════════════════════════════════════
# Central coordination agent for the MAAI-SkinDx pipeline.
# Implements the Taseneem Adaptive Coordination Intelligence
# framework (Phases 1 through 9) and aligns with the Strong
# Coordination Agent Evaluation Guide.
#
# No agent communicates directly with another agent.
# All inter-agent communication passes through this Orchestrator.
#
# Coordination phases implemented:
#
#   Phase 1  — Unified message format
#              Every agent output standardised with confidence,
#              uncertainty, quality score, latency, availability.
#              Agent-specific confidence values:
#                Agent 1   : 1.0 when PASS (binary quality check)
#                Agent 2   : 1.0 when PASS (feature extraction)
#                Agent 3   : top-1 softmax probability
#                Agent 4   : 0.85 focused / 0.60 diffuse focus
#                Clinical  : RandomForest ML probability
#                Agent 5   : chunks retrieved / 5 (capped at 1.0)
#
#   Phase 2  — Agent assessment
#              Base trust scores derived from validated metrics
#
#   Phase 4  — Adaptive trust
#              Per-case trust adapted by quality, uncertainty,
#              and contextual relevance
#
#   Phase 5  — Adaptive fusion
#              Weighted formula: w_i = T_i x Q_i x (1-U_i) x R_i
#              Normalised so weights sum to 1.
#              Fused malignancy signal corrected:
#                Malignant class predicted: signal = confidence
#                Benign class predicted:    signal = 1 - confidence
#
#   Phase 6  — Pre-fusion and post-fusion reasoning
#              Pre:  assess each agent independently
#              Post: interpret combined evidence sufficiency
#
#   Phase 7  — Conflict resolution
#              Detect and classify image vs symptom disagreement
#              Vote entropy computed per case
#
#   Phase 8  — Decision intelligence
#              Confidence-aware clinical decision policy
#              Seven decision types with urgency levels
#              Includes REQUEST_EVIDENCE path
#
#   Phase 9  — Action intelligence
#              Agent 7 Expert generates full clinical report
#
# Pipeline flow:
#   Agent 1 — Image quality check
#   Agent 2 — Visual feature extraction
#   Agent 3 — Disease classification
#   OOD confidence check
#   Phase 1 unified message wrapping
#   Phase 2+4 agent assessment and adaptive trust
#   Agent 4 + Clinical Agent in parallel
#   Phase 5 adaptive fusion (corrected malignancy signal)
#   Phase 6 pre-fusion reasoning
#   Phase 7 conflict resolution
#   Phase 6 post-fusion reasoning
#   Phase 8 decision intelligence
#   Agent 5 RAG knowledge retrieval
#   Agent 7 Expert clinical report generation
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

# ── Phase 2 — Base trust scores ────────────────────────────────────
# Derived from empirically validated performance metrics.
# Agent 1 : F1 97.89% on skin detection test set
# Agent 2 : Zero failures across all test cases
# Agent 3 : Macro F1 74.18% on 7,734 hold-out images
# Agent 4 : Qualitative XAI — moderate reliability
# Clinical : CV F1 92.68% on PAD-UFES-20
# Agent 5 : RAG retrieval — moderate reliability
BASE_TRUST_SCORES = {
    "agent1"  : 0.979,
    "agent2"  : 0.950,
    "agent3"  : 0.742,
    "agent4"  : 0.850,
    "clinical": 0.927,
    "agent5"  : 0.800,
}

# OOD confidence threshold
# In-distribution cases score 65-73%
# Out-of-distribution cases score 35-40%
OOD_THRESHOLD = 0.40

# Malignant class indices in Agent 3 taxonomy
# 0 = Melanoma, 2 = BCC, 3 = AK/SCC
MALIGNANT_CLASSES = {0, 2, 3}

# Trust evolution log
os.makedirs(
    os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
TRUST_LOG_PATH = os.path.join(
    PROJECT_ROOT, "logs", "trust_evolution.json")


class Orchestrator:
    """
    Adaptive Coordination Intelligence Agent for MAAI-SkinDx.

    Implements the full Taseneem framework coordination pipeline.
    Serves as the sole communication hub between all specialist
    agents. No agent in the system holds a reference to any
    other agent.
    """

    def __init__(self):
        print("\n" + "=" * 60)
        print("  MAAI-SkinDx | Orchestrator")
        print("  Adaptive Coordination Intelligence")
        print("  Taseneem Framework Phases 1 to 9")
        print("=" * 60)

        print("\n  Loading agents...")
        self.agent1   = ImageQualityAgent()
        self.agent2   = FeatureExtractionAgent()
        self.agent3   = DiagnosisAgent()
        self.agent4   = XAIAgent()
        self.clinical = ClinicalAgent()
        self.agent5   = RAGAgent()

        # Agent 7 Expert
        self.agent7 = None
        try:
            from agents.agent7_Expert import ClinicalExpertAgent
            self.agent7 = ClinicalExpertAgent()
            print("  Agent 7 Expert loaded.")
        except Exception as e:
            print(f"  Agent 7 Expert not available: {e}")

        # Mutable trust scores
        # Updated per-case by Phase 4 adaptive trust
        # Long-term: Agent 6 feedback updates base scores
        self.trust_scores  = BASE_TRUST_SCORES.copy()
        self.trust_history = []

        print("\n  Base trust scores:")
        for name, score in self.trust_scores.items():
            print(f"    {name:<12}: {score:.3f}")
        print("=" * 60 + "\n")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1 — Unified Message Format
    # Standardise every agent output into a consistent structure.
    # Agent-specific confidence values are provided explicitly
    # because each agent produces a different type of output.
    # ══════════════════════════════════════════════════════════════

    def _wrap(
        self,
        agent_name : str,
        raw_result : dict,
        start_time : float,
        confidence : float = None,
    ) -> dict:
        """
        Phase 1: Wrap raw agent output into unified message format.

        Confidence values per agent:
          Agent 1   : 1.0 when PASS (binary — image is skin or not)
          Agent 2   : 1.0 when PASS (feature extraction either works or fails)
          Agent 3   : top-1 softmax probability from classifier
          Agent 4   : 0.85 focused, 0.60 diffuse (Grad-CAM quality)
          Clinical  : RandomForest ML malignancy probability
          Agent 5   : chunks retrieved divided by 5, capped at 1.0
          Agent 7   : 1.0 when PASS (report generated successfully)

        Uncertainty is computed as 1 minus confidence.
        Quality score is derived from status.
        """
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
        """
        Phase 2 and 4: Agent assessment and adaptive trust.

        Adaptive trust formula (Taseneem Phase 4):
          T_adaptive = T_base x Q x (1 - U) x R

        Where:
          T_base : validated base trust score
          Q      : data quality score for this case
          U      : prediction uncertainty for this case
          R      : contextual relevance for this case

        Contextual relevance is case-specific:
          Clinical: 1.0 with symptoms, 0.3 without
          Agent 4 : min(1.0, agent3_confidence + 0.2)
          Others  : 1.0
        """
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

            # Contextual relevance
            if name == "clinical":
                relevance = 1.0 if has_symptoms else 0.3
            elif name == "agent4":
                relevance = min(1.0, a3_conf + 0.2)
            else:
                relevance = 1.0

            # Adaptive trust
            t_adaptive = (
                base_trust *
                quality *
                (1.0 - uncertainty) *
                relevance
            )
            t_adaptive = round(
                min(1.0, max(0.0, t_adaptive)), 4)

            assessment[name] = {
                "base_trust"    : base_trust,
                "quality"       : quality,
                "uncertainty"   : uncertainty,
                "relevance"     : round(relevance, 3),
                "adaptive_trust": t_adaptive,
            }
            trust_scores[name] = t_adaptive

        # Log trust evolution for Phase 4 tracking
        self.trust_history.append({
            "timestamp"   : datetime.now().isoformat(),
            "trust_scores": trust_scores.copy(),
        })

        return assessment, trust_scores

    # ══════════════════════════════════════════════════════════════
    # PHASE 5 — Adaptive Fusion
    # Combine agent evidence using normalised adaptive trust.
    #
    # CRITICAL FIX — Fused malignancy signal:
    # Agent 3 confidence is a class probability, not a malignancy
    # probability. When Agent 3 predicts a BENIGN class at 67.6%
    # confidence, the image malignancy signal is LOW (0.324),
    # not HIGH (0.676). The correct formula is:
    #   Malignant class  : image_signal = confidence
    #   Benign class     : image_signal = 1.0 - confidence
    # ══════════════════════════════════════════════════════════════

    def _adaptive_fusion(
        self,
        trust_scores          : dict,
        a3_confidence         : float,
        ml_probability        : float,
        predicted_is_malignant: bool,
    ) -> dict:
        """
        Phase 5: Adaptive fusion with corrected malignancy signal.

        Normalises adaptive trust scores so they sum to 1.
        Computes a fused malignancy signal from image evidence
        and symptom evidence weighted by normalised trust.

        The fused signal correctly reflects:
          If Agent 3 predicts melanoma at 0.85 confidence,
          image malignancy signal = 0.85 (high threat)

          If Agent 3 predicts benign nevus at 0.67 confidence,
          image malignancy signal = 0.33 (low threat from image)
        """
        total = sum(trust_scores.values())
        if total <= 0:
            total = 1.0

        weights = {
            name: round(score / total, 4)
            for name, score in trust_scores.items()
        }

        # Normalise image vs clinical weights
        w3    = weights.get("agent3", 0.0)
        wc    = weights.get("clinical", 0.0)
        w_sum = w3 + wc
        if w_sum <= 0:
            w_sum = 1.0

        w3_norm = w3 / w_sum
        wc_norm = wc / w_sum

        # Correct image malignancy signal
        if predicted_is_malignant:
            image_mal_signal = a3_confidence
        else:
            image_mal_signal = 1.0 - a3_confidence

        # Clinical malignancy signal
        ml_signal = (
            ml_probability
            if ml_probability is not None and ml_probability >= 0
            else 0.0
        )

        fused_malignancy = round(
            (image_mal_signal * w3_norm) +
            (ml_signal * wc_norm),
            4
        )

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
        """
        Phase 6 (pre): Assess each evidence source independently.

        Evaluates image confidence level, evidence strength,
        clinical data availability, malignancy flag,
        prediction entropy, and trust-weighted confidence.
        """
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

        # Prediction entropy
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

        # Trust-weighted confidence
        a3_assess = assessment.get("agent3", {})
        pf["weighted_confidence"] = round(
            confidence * a3_assess.get("adaptive_trust", 0.742),
            4)

        # Evidence strength labels
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
              f"{pf['image_confidence_level']} "
              f"({confidence*100:.1f}%)")
        print(f"  Image evidence      : "
              f"{pf['image_evidence_strength']}")
        print(f"  Clinical evidence   : "
              f"{pf['clinical_evidence_strength']}")
        print(f"  Malignant predicted : "
              f"{pf['image_says_malignant']}")
        print(f"  Prediction entropy  : "
              f"{pf['prediction_entropy']:.4f} "
              f"({'HIGH' if pf['high_entropy'] else 'NORMAL'})")
        print(f"  Weighted confidence : "
              f"{pf['weighted_confidence']:.4f}")

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
        """
        Phase 7: Conflict resolution algorithm.

        Detects and classifies disagreement between image
        and symptom evidence. Vote entropy is computed to
        quantify severity of disagreement.

        Conflict types:
          IMAGE_LOW_SYMPTOM_HIGH : Most dangerous
          IMAGE_HIGH_SYMPTOM_LOW : Requires review
          FUSED_SIGNAL_HIGH      : Fusion suggests malignancy
          LOW_CONFIDENCE         : Insufficient image evidence
          NO_CLINICAL_DATA       : No symptoms submitted
          CONCORDANT             : Agents agree
        """
        conflict = {
            "type"        : "CONCORDANT",
            "severity"    : "NONE",
            "description" : (
                "Image and clinical evidence are in agreement. "
                "No discordance detected."
            ),
            "resolved_by" : "CONCORDANT",
            "vote_entropy": 0.0,
        }

        if not clinical_result or clinical_result.get(
                "status") == "SKIP":
            conflict["type"]        = "NO_CLINICAL_DATA"
            conflict["description"] = (
                "No symptom data was submitted. "
                "The diagnostic decision is based on "
                "image evidence only.")
            return conflict

        ml_prob    = clinical_result.get("ml_probability")
        risk_level = clinical_result.get("risk_level", "LOW")
        risk_score = clinical_result.get("risk_score", 0)

        image_malignant   = predicted_index in MALIGNANT_CLASSES
        symptom_malignant = (
            (ml_prob is not None and ml_prob > 0.60) or
            risk_level in ("HIGH", "CRITICAL")
        )
        fused_mal = fusion.get("fused_malignancy", 0.0)

        # Vote entropy between image and clinical
        p_image    = confidence if image_malignant else (
            1.0 - confidence)
        p_clinical = ml_prob if ml_prob is not None else 0.5
        p_avg      = (p_image + p_clinical) / 2.0
        if 0 < p_avg < 1:
            vote_entropy = round(
                -p_avg * math.log(p_avg + 1e-10)
                - (1 - p_avg) * math.log(
                    1 - p_avg + 1e-10),
                4)
        else:
            vote_entropy = 0.0
        conflict["vote_entropy"] = vote_entropy

        if not image_malignant and symptom_malignant:
            ml_pct = (
                f"{ml_prob*100:.1f}%"
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
                    f"Clinical evidence overrides image prediction."
                ),
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
                    f"Image prediction requires clinical review."
                ),
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
                    f"warrants clinical review."
                ),
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
                    f"for a reliable diagnostic decision."
                ),
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
        """
        Phase 6 (post): Interpret combined evidence after fusion.

        Determines whether evidence is sufficient for a decision
        and classifies the fused malignancy signal level.
        """
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
                "A reliable diagnostic decision cannot be made."
            )

        elif (conflict_type == "CONCORDANT"
              and img_str == "STRONG"):
            pfr["evidence_sufficient"] = True
            pfr["evidence_quality"]    = "STRONG"
            pfr["evidence_assessment"] = (
                "Image and clinical evidence agree strongly. "
                "Combined evidence is sufficient to support "
                "a diagnostic decision."
            )

        elif conflict_type in (
                "IMAGE_LOW_SYMPTOM_HIGH", "FUSED_SIGNAL_HIGH"):
            pfr["evidence_sufficient"] = True
            pfr["evidence_quality"]    = "CONFLICTED"
            pfr["evidence_assessment"] = (
                "Clinical evidence overrides image prediction. "
                "Conflict detected and resolved. "
                "Escalation is indicated."
            )

        elif conflict_type == "LOW_CONFIDENCE":
            pfr["evidence_sufficient"] = False
            pfr["evidence_quality"]    = "WEAK"
            pfr["evidence_assessment"] = (
                "Image evidence is insufficient. "
                "Clinical assessment or dermoscopy is required."
            )

        else:
            pfr["evidence_sufficient"] = True
            pfr["evidence_quality"]    = "ADEQUATE"
            pfr["evidence_assessment"] = (
                "Combined evidence is adequate to support "
                "a provisional diagnostic decision."
            )

        # Fused signal level
        if fused_mal > 0.75:
            pfr["fused_signal_level"] = "HIGH"
        elif fused_mal > 0.50:
            pfr["fused_signal_level"] = "MODERATE"
        elif fused_mal > 0.25:
            pfr["fused_signal_level"] = "LOW"
        else:
            pfr["fused_signal_level"] = "BENIGN"

        print(f"\n  [Post-Fusion Reasoning]")
        print(f"  Evidence quality    : "
              f"{pfr['evidence_quality']}")
        print(f"  Evidence sufficient : "
              f"{pfr['evidence_sufficient']}")
        print(f"  Fused signal level  : "
              f"{pfr['fused_signal_level']} "
              f"({fused_mal:.4f})")

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
    ) -> dict:
        """
        Phase 8: Confidence-aware clinical decision intelligence.

        Decision types:
          OOD_WARNING      : Image outside trained distribution
          ESCALATE         : IMAGE_LOW_SYMPTOM_HIGH
          REVIEW           : IMAGE_HIGH_SYMPTOM_LOW
          REQUEST_EVIDENCE : Insufficient evidence
          UNCERTAIN        : Low confidence
          REFER_URGENT     : Concordant CRITICAL risk
          REFER            : Concordant HIGH risk
          CONCORDANT       : Concordant MODERATE risk
          ROUTINE          : Concordant LOW risk
        """
        conflict_type = conflict.get("type", "CONCORDANT")
        risk_level    = (
            clinical_result.get("risk_level")
            if clinical_result and
            clinical_result.get("status") == "PASS"
            else None
        )
        evidence_ok = postfusion.get("evidence_sufficient", True)
        fused_mal   = fusion.get("fused_malignancy", 0.0)

        if is_ood:
            return {
                "decision" : "OOD_WARNING",
                "urgency"  : "REFER_SPECIALIST",
                "confidence": "LOW",
                "rationale": (
                    "The submitted image has a top-1 confidence "
                    "below the reliable threshold of 0.40. "
                    "The image may depict a condition outside "
                    "the nine trained dermatological classes "
                    "such as chickenpox or monkeypox. "
                    "A reliable AI diagnosis cannot be provided. "
                    "Direct specialist consultation is required."
                ),
            }

        if conflict_type == "IMAGE_LOW_SYMPTOM_HIGH":
            return {
                "decision" : "ESCALATE",
                "urgency"  : "URGENT",
                "confidence": "HIGH",
                "rationale": (
                    "Critical discordance detected. The image "
                    "prediction indicates a benign condition "
                    "but patient symptom evidence strongly "
                    "suggests elevated malignancy risk. "
                    "Clinical symptom evidence overrides the "
                    "image prediction. Urgent specialist "
                    "referral is required."
                ),
            }

        if conflict_type in (
                "IMAGE_HIGH_SYMPTOM_LOW", "FUSED_SIGNAL_HIGH"):
            return {
                "decision" : "REVIEW",
                "urgency"  : "WITHIN_1_WEEK",
                "confidence": "MODERATE",
                "rationale": (
                    "Moderate discordance detected between "
                    "image and clinical evidence. Clinical "
                    "review and dermoscopic examination "
                    "within one week is recommended."
                ),
            }

        if not evidence_ok and conflict_type == "LOW_CONFIDENCE":
            return {
                "decision" : "REQUEST_EVIDENCE",
                "urgency"  : "WITHIN_4_WEEKS",
                "confidence": "LOW",
                "rationale": (
                    "Image classification confidence is "
                    "insufficient to support a reliable "
                    "diagnostic decision. Additional evidence "
                    "is required: dermoscopic examination, "
                    "patient symptom history, or a higher "
                    "quality image submission."
                ),
            }

        if not evidence_ok:
            return {
                "decision" : "UNCERTAIN",
                "urgency"  : "WITHIN_4_WEEKS",
                "confidence": "LOW",
                "rationale": (
                    "Combined evidence quality is insufficient "
                    "for a confident diagnostic decision. "
                    "Clinical assessment is recommended."
                ),
            }

        if risk_level == "CRITICAL":
            return {
                "decision" : "REFER_URGENT",
                "urgency"  : "URGENT",
                "confidence": "HIGH",
                "rationale": (
                    "Image and clinical evidence are concordant. "
                    "Clinical risk level is CRITICAL. "
                    "Urgent specialist referral is required."
                ),
            }

        if risk_level == "HIGH":
            return {
                "decision" : "REFER",
                "urgency"  : "WITHIN_1_WEEK",
                "confidence": "HIGH",
                "rationale": (
                    "Image and clinical evidence are concordant. "
                    "Clinical risk level is HIGH. "
                    "Specialist referral within one week."
                ),
            }

        if risk_level == "MODERATE" or fused_mal > 0.40:
            return {
                "decision" : "CONCORDANT",
                "urgency"  : "WITHIN_4_WEEKS",
                "confidence": "MODERATE",
                "rationale": (
                    "Image and clinical evidence are concordant. "
                    "Clinical risk is MODERATE. "
                    "Clinical assessment within four weeks."
                ),
            }

        return {
            "decision" : "ROUTINE",
            "urgency"  : "ROUTINE",
            "confidence": "HIGH",
            "rationale": (
                "Image and clinical evidence are concordant. "
                "Clinical risk is LOW. "
                "Routine monitoring is recommended."
            ),
        }

    # ══════════════════════════════════════════════════════════════
    # MAIN PIPELINE
    # ══════════════════════════════════════════════════════════════

    def process(
        self,
        image_path : str,
        symptoms   : dict = None,
    ) -> dict:
        """
        Main entry point. Process one patient case end-to-end.
        Implements Taseneem framework Phases 1 through 9.
        """
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
        t1 = time.time()
        agent1_result = self.agent1.run(image_path)

        # Phase 1: Agent 1 confidence = 1.0 when PASS
        # The quality check is binary — image is valid or not
        a1_conf = (
            1.0 if agent1_result["status"] == "PASS" else 0.0)
        w1 = self._wrap(
            "agent1", agent1_result, t1, confidence=a1_conf)
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
            self._print_summary(result)
            return result

        print(f"  Agent 1 -- PASS | "
              f"conf={a1_conf:.2f} | "
              f"{w1['latency_s']}s")

        # ── Agent 2 ────────────────────────────────────────────────
        print("\n-- Step 2: Agent 2 -- Feature Extraction --")
        t2 = time.time()
        agent2_result = self.agent2.run(image_path)

        # Phase 1: Agent 2 confidence = 1.0 when PASS
        # Feature extraction either succeeds completely or fails
        a2_conf = (
            1.0 if agent2_result["status"] == "PASS" else 0.0)
        w2 = self._wrap(
            "agent2", agent2_result, t2, confidence=a2_conf)
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
            self._print_summary(result)
            return result

        feature_vector = agent2_result.get("feature_vector")
        feature_dim    = agent2_result.get("feature_dim", 0)
        print(f"  Agent 2 -- PASS | "
              f"conf={a2_conf:.2f} | "
              f"{feature_dim} dims | "
              f"{w2['latency_s']}s")

        # ── Agent 3 ────────────────────────────────────────────────
        print("\n-- Step 3: Agent 3 -- Classification --")
        t3 = time.time()
        agent3_result = self.agent3.run(
            image_path      = image_path,
            agent2_features = feature_vector,
        )

        # Phase 1: Agent 3 confidence = top-1 softmax probability
        confidence = agent3_result.get("confidence", 0.0)
        w3 = self._wrap(
            "agent3", agent3_result, t3, confidence=confidence)
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
            self._print_summary(result)
            return result

        predicted_name  = agent3_result.get(
            "predicted_name",  "Unknown")
        predicted_class = agent3_result.get(
            "predicted_class", "Unknown")
        predicted_index = agent3_result.get("predicted_index", 0)
        confidence_pct  = confidence * 100
        is_malignant    = predicted_index in MALIGNANT_CLASSES

        print(f"  Agent 3 -- PASS | "
              f"{predicted_name} ({confidence_pct:.1f}%) | "
              f"{'MALIGNANT' if is_malignant else 'benign'} | "
              f"{w3['latency_s']}s")

        # ── OOD Check ──────────────────────────────────────────────
        print(f"\n-- OOD Check --")
        is_ood = confidence < OOD_THRESHOLD
        if is_ood:
            print(f"  WARNING: {confidence_pct:.1f}% below "
                  f"threshold {OOD_THRESHOLD*100:.0f}%")
            print(f"  Possible out-of-distribution image")
        else:
            print(f"  In-distribution confirmed: "
                  f"{confidence_pct:.1f}%")

        # ── Parallel: Agent 4 + Clinical ───────────────────────────
        print(
            "\n-- Step 4+5: Agent 4 + Clinical [parallel] --")

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
                        agent4_result = {
                            "status": "FAIL",
                            "reason": str(e)}
                    else:
                        clinical_result = {
                            "status": "FAIL",
                            "reason": str(e)}

        result["agents"]["agent4"]   = agent4_result
        result["agents"]["clinical"] = clinical_result

        # ML probability from Clinical Agent
        ml_prob = (
            clinical_result.get("ml_probability")
            if clinical_result and
            clinical_result.get("status") == "PASS"
            else None
        )

        # Phase 1: Agent 4 confidence based on focus quality
        # Focused heatmap = 0.85, diffuse = 0.60, fail = 0.0
        focus_stats   = (agent4_result or {}).get(
            "focus_stats") or {}
        focus_type_val = focus_stats.get("focus_type", "")
        if (agent4_result or {}).get("status") == "PASS":
            a4_conf = (
                0.85
                if focus_type_val in (
                    "focus_on_lesion", "focus_on_border")
                else 0.60
            )
        else:
            a4_conf = 0.0

        w4 = self._wrap(
            "agent4",
            agent4_result or {},
            t45,
            confidence=a4_conf,
        )

        # Phase 1: Clinical confidence = ML probability
        wc = self._wrap(
            "clinical",
            {**(clinical_result or {}),
             "confidence": ml_prob or 0.0},
            t45,
            confidence=ml_prob or 0.0,
        )

        result["unified_outputs"]["agent4"]   = w4
        result["unified_outputs"]["clinical"] = wc

        # ── Phase 2+4 — Assessment and Adaptive Trust ──────────────
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

        # ── Phase 5 — Adaptive Fusion ──────────────────────────────
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
        print(f"  Fused malignancy signal : "
              f"{fusion['fused_malignancy']:.4f}")

        # ── Phase 6 — Pre-Fusion Reasoning ────────────────────────
        print(f"\n-- Phase 6: Pre-Fusion Reasoning --")
        prefusion = self._pre_fusion_reasoning(
            agent3_result   = agent3_result,
            confidence      = confidence,
            predicted_index = predicted_index,
            symptoms        = symptoms,
            assessment      = assessment,
        )
        result["prefusion"] = prefusion

        # ── Phase 7 — Conflict Resolution ─────────────────────────
        print(f"\n-- Phase 7: Conflict Resolution --")
        conflict = self._conflict_resolution(
            predicted_index = predicted_index,
            confidence      = confidence,
            clinical_result = clinical_result,
            prefusion       = prefusion,
            fusion          = fusion,
        )
        result["conflict"] = conflict

        # ── Phase 6 — Post-Fusion Reasoning ───────────────────────
        print(f"\n-- Phase 6: Post-Fusion Reasoning --")
        postfusion = self._post_fusion_reasoning(
            conflict  = conflict,
            fusion    = fusion,
            prefusion = prefusion,
            is_ood    = is_ood,
        )
        result["postfusion"] = postfusion

        # ── Phase 8 — Decision Intelligence ───────────────────────
        print(f"\n-- Phase 8: Decision Intelligence --")
        decision = self._decision_intelligence(
            conflict        = conflict,
            prefusion       = prefusion,
            postfusion      = postfusion,
            fusion          = fusion,
            clinical_result = clinical_result,
            is_ood          = is_ood,
        )
        result["decision"] = decision
        print(f"  Decision   : {decision['decision']}")
        print(f"  Urgency    : {decision['urgency']}")
        print(f"  Confidence : {decision['confidence']}")

        # ── Extract outputs for downstream agents ──────────────────
        focus_type      = focus_stats.get("focus_type")
        xai_explanation = (
            (agent4_result or {}).get("explanation")
            if (agent4_result or {}).get("status") == "PASS"
            else None
        )
        risk_level   = None
        risk_score   = None
        risk_factors = []

        if clinical_result and clinical_result.get(
                "status") == "PASS":
            risk_level   = clinical_result.get("risk_level")
            risk_score   = clinical_result.get("risk_score")
            risk_factors = clinical_result.get(
                "risk_factors", [])

        # ── Agent 5 — RAG Knowledge Retrieval ─────────────────────
        print("\n-- Step 6: Agent 5 -- RAG Knowledge --")
        t5 = time.time()
        agent5_result = self.agent5.run(
            case_id         = case_id,
            predicted_index = predicted_index,
            predicted_name  = predicted_name,
            confidence      = confidence,
            top_5           = agent3_result.get(
                "top_5_predictions", []),
            risk_level      = risk_level,
            risk_score      = risk_score,
            ml_probability  = ml_prob,
            focus_type      = focus_type,
            xai_explanation = xai_explanation,
        )

        # Phase 1: Agent 5 confidence = chunks/5 capped at 1.0
        chunks  = agent5_result.get("chunks_retrieved", 0)
        a5_conf = (
            min(1.0, chunks / 5.0)
            if agent5_result.get("status") == "PASS"
            else 0.0
        )
        w5 = self._wrap(
            "agent5", agent5_result, t5, confidence=a5_conf)
        result["agents"]["agent5"]          = agent5_result
        result["unified_outputs"]["agent5"] = w5

        print(f"  Agent 5 -- {agent5_result.get('status')} | "
              f"conf={a5_conf:.2f} | "
              f"{chunks} chunks | "
              f"{w5['latency_s']}s")

        # ── Phase 9 — Agent 7 Expert — Clinical Report ─────────────
        agent7_result = None
        if self.agent7 is not None:
            print(
                "\n-- Step 7: Agent 7 Expert -- Clinical Report --")
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
                conflict_type      = conflict.get(
                    "type", "CONCORDANT"),
                conflict_severity  = conflict.get(
                    "severity", "NONE"),
                conflict_desc      = conflict.get(
                    "description", ""),
                decision           = decision.get(
                    "decision", "ROUTINE"),
                urgency            = decision.get(
                    "urgency", "ROUTINE"),
                decision_rationale = decision.get(
                    "rationale", ""),
                symptoms           = symptoms,
                fusion_weights     = fusion.get("weights", {}),
                fused_malignancy   = fusion.get(
                    "fused_malignancy", 0.0),
            )

            # Phase 1: Agent 7 confidence = 1.0 when PASS
            a7_conf = (
                1.0
                if (agent7_result or {}).get("status") == "PASS"
                else 0.0
            )
            w7 = self._wrap(
                "agent7",
                agent7_result or {},
                t7,
                confidence=a7_conf,
            )
            result["agents"]["agent7"]          = agent7_result
            result["unified_outputs"]["agent7"] = w7
            print(f"  Agent 7 -- "
                  f"{(agent7_result or {}).get('status','UNKNOWN')} "
                  f"| conf={a7_conf:.2f} | "
                  f"{w7['latency_s']}s")

        # ── Build final result ─────────────────────────────────────
        elapsed  = round(time.time() - start_time, 2)
        ood_msg  = (
            " OOD WARNING: image may be outside trained classes."
            if is_ood else "")

        result.update({
            "pipeline_status"  : "COMPLETE",
            "final_status"     : "PASS",
            "predicted_disease": predicted_name,
            "confidence"       : f"{confidence_pct:.1f}%",
            "is_ood"           : is_ood,
            "is_malignant"     : is_malignant,
            "top_5_predictions": agent3_result.get(
                "top_5_predictions", []),
            "clinical_risk"    : risk_level,
            "rag_summary"      : agent5_result.get("summary"),
            "elapsed_seconds"  : elapsed,
            "message"          : (
                f"Pipeline complete. "
                f"Diagnosis: {predicted_name} "
                f"({confidence_pct:.1f}% | "
                f"{'MALIGNANT' if is_malignant else 'benign'})."
                f"{ood_msg} "
                f"Conflict: {conflict.get('type')}. "
                f"Decision: {decision.get('decision')}. "
                f"Urgency: {decision.get('urgency')}. "
                f"Time: {elapsed}s."
            ),
        })

        self._print_summary(result)
        return result

    # ══════════════════════════════════════════════════════════════
    # SUMMARY PRINTER
    # ══════════════════════════════════════════════════════════════

    def _print_summary(self, result: dict):
        """Print complete coordination intelligence summary."""
        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  Pipeline Summary -- {result['case_id']}")
        print(sep)
        print(f"  Status : {result['pipeline_status']}")
        print(f"  Time   : {result['elapsed_seconds']}s")
        print(f"  Message: {result['message']}")

        # Phase 1 — Unified agent status
        print(f"\n  Agent Status (Phase 1 Unified Format):")
        for name, wo in result.get(
                "unified_outputs", {}).items():
            a     = result.get("assessment", {}).get(name, {})
            trust = (a.get("adaptive_trust")
                    if a
                    else BASE_TRUST_SCORES.get(name, 0.0))
            label = "adaptive" if a else "base"
            print(
                f"  {name.upper():<12}: "
                f"{wo.get('status','?'):<6} | "
                f"conf={wo.get('confidence',0):.3f} | "
                f"unc={wo.get('uncertainty',0):.3f} | "
                f"trust={trust:.3f} ({label}) | "
                f"{wo.get('latency_s',0):.2f}s"
            )

        # OOD
        if result.get("is_ood"):
            print(f"\n  OOD WARNING: confidence below "
                  f"{OOD_THRESHOLD*100:.0f}%")

        # Top 5
        if result.get("top_5_predictions"):
            print(f"\n  Top 5 Predictions:")
            for p in result["top_5_predictions"]:
                bar = "=" * int(p["confidence"] * 20)
                print(f"    {p['rank']}. "
                      f"{p['class_name']:<35} "
                      f"{p['confidence_pct']:>6}  {bar}")

        # Phase 5 — Fusion
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

        # Phase 6 — Pre-fusion
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

        # Phase 6 — Post-fusion
        pfr = result.get("postfusion", {})
        if pfr:
            print(f"\n  Phase 6 -- Post-Fusion Reasoning:")
            print(f"    Evidence quality  : "
                  f"{pfr.get('evidence_quality','-')}")
            print(f"    Fused signal      : "
                  f"{pfr.get('fused_signal_level','-')}")

        # Phase 7 — Conflict
        conflict = result.get("conflict", {})
        if conflict:
            print(f"\n  Phase 7 -- Conflict Resolution:")
            print(f"    Type        : {conflict.get('type','-')}")
            print(f"    Severity    : "
                  f"{conflict.get('severity','-')}")
            print(f"    Vote entropy: "
                  f"{conflict.get('vote_entropy',0):.4f}")
            print(f"    Resolved by : "
                  f"{conflict.get('resolved_by','-')}")

        # Phase 8 — Decision
        dec = result.get("decision", {})
        if dec:
            print(f"\n  Phase 8 -- Decision Intelligence:")
            print(f"    Decision    : {dec.get('decision','-')}")
            print(f"    Urgency     : {dec.get('urgency','-')}")
            print(f"    Confidence  : {dec.get('confidence','-')}")
            rat = dec.get("rationale", "")
            if rat:
                print(f"    Rationale   : {rat[:90]}...")

        # Clinical
        clin = result.get("agents", {}).get("clinical") or {}
        if clin.get("risk_level"):
            print(f"\n  Clinical Assessment:")
            print(f"    Risk level  : {clin['risk_level']} "
                  f"(score {clin.get('risk_score',0)}/100)")
            if clin.get("ml_probability") is not None:
                print(f"    ML prob     : "
                      f"{clin['ml_probability']*100:.1f}%")
            print(f"    Recommend   : "
                  f"{clin.get('recommendation','')}")

        # RAG
        rag = result.get("agents", {}).get("agent5") or {}
        if rag.get("status") == "PASS":
            print(f"\n  Agent 5 -- RAG Knowledge:")
            print(f"    Chunks : "
                  f"{rag.get('chunks_retrieved',0)}")
            if rag.get("summary"):
                print(f"    Summary:")
                for line in rag["summary"].split(". "):
                    if line.strip():
                        print(f"      {line.strip()}.")

        # Phase 9 — Agent 7
        a7 = result.get("agents", {}).get("agent7") or {}
        if a7.get("status") == "PASS":
            print(f"\n  Phase 9 -- Agent 7 Expert Report:")
            print(f"    Status : PASS")
            print(f"    Saved  : "
                  f"{a7.get('report_path','')}")

        print(f"{sep}\n")