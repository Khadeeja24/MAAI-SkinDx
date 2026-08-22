# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Clinical Agent — Symptom Risk Assessment
# ══════════════════════════════════════════════════════════════════
# Runs IN PARALLEL with Agent 4 after Agent 3 finishes.
# Reports results back to Orchestrator only.
#
# Takes patient symptom form and produces:
#   1. Rule-based risk score (0-100) — fully explainable
#   2. ML model probability — RandomForest on PAD-UFES-20
#   3. Risk level — LOW / MODERATE / HIGH / CRITICAL
#   4. Risk factors — which factors contributed and by how much
#   5. Clinical recommendation — what the doctor should do
#
# Fixes applied:
#   1. Diameter zero bug — explicit None check.
#      0 is a valid lesion diameter. Using (val or median) would
#      silently replace 0 with the median because 0 is falsy.
#   2. Age validated and clipped to 1-100
#   3. Fitzpatrick validated and clipped to 1-6
#   4. Full try/except in _ml_probability — bad input field
#      cannot crash the agent; falls back to -1.0 gracefully
#   5. import traceback for production debugging
#   6. UTF-8 encoding on all file writes
#
# Never sees the image. Never talks to Agent 4 directly.
# ══════════════════════════════════════════════════════════════════

import os
import sys
import pickle
import traceback
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import (
    MODEL_PATH, OUTPUT_DIR,
    RISK_WEIGHTS, RISK_LEVELS,
    RISK_RECOMMENDATIONS, ALL_FEATURES
)


class ClinicalAgent:

    def __init__(self):
        self.name       = "Clinical Agent — Symptom Risk Assessment"
        self.model      = None
        self.model_data = None
        print(f"\n[{self.name}] Initialising ...")
        self._load_model()
        print(f"[{self.name}] Ready")

    # ─── Model Loading ─────────────────────────────────────────────

    def _load_model(self):
        """Load trained RandomForest model from disk."""
        if not os.path.exists(MODEL_PATH):
            print(f"  [Clinical] WARNING: Model not found")
            print(f"  Run: python scripts/train_clinical_model.py")
            return
        try:
            with open(MODEL_PATH, "rb") as f:
                self.model_data = pickle.load(f)
            self.model = self.model_data["model"]
            print(f"  [Clinical] Model loaded")
            print(f"  [Clinical] CV F1    : "
                  f"{self.model_data['cv_f1_mean']*100:.2f}%")
            print(f"  [Clinical] CV AUC   : "
                  f"{self.model_data['cv_auc_mean']:.4f}")
            print(f"  [Clinical] Features : "
                  f"{len(self.model_data['features'])}")
        except Exception as e:
            print(f"  [Clinical] Load failed: {e}")
            traceback.print_exc()
            self.model = None

    # ─── Encoding Helpers ──────────────────────────────────────────

    def _encode_symptom(self, val) -> float:
        """
        Encode patient-reported symptom.
        Matches encoding used in train_clinical_model.py exactly.
          yes / true / 1     → 1.0  (symptom present)
          no  / false / 0    → 0.0  (symptom absent)
          unk / None / other → 0.5  (unknown)
        """
        if val is None:
            return 0.5
        s = str(val).strip().lower()
        if s in ("yes", "true", "1"):
            return 1.0
        elif s in ("no", "false", "0"):
            return 0.0
        else:
            return 0.5

    def _encode_bool(self, val) -> float:
        """
        Encode yes/no history fields.
        None → 0.0 (assume no history when not provided)
        """
        if val is None:
            return 0.0
        return 1.0 if str(val).strip().lower() in \
               ("yes", "true", "1") else 0.0

    # ─── Rule-based Scoring ────────────────────────────────────────

    def _rule_based_score(self, symptoms: dict) -> tuple:
        """
        Compute evidence-based risk score 0-100.
        Every point is traceable to a specific clinical factor.
        Fully transparent — no black box.

        Returns:
            score   (int)  : total score capped at 100
            factors (list) : triggered risk factor descriptions
        """
        score   = 0
        factors = []

        # Lesion behaviour — strongest clinical predictors
        if self._encode_symptom(symptoms.get("changed")) == 1.0:
            score += RISK_WEIGHTS["changed"]
            factors.append(
                f"Lesion changed recently "
                f"(+{RISK_WEIGHTS['changed']} pts)")

        if self._encode_symptom(symptoms.get("bleed")) == 1.0:
            score += RISK_WEIGHTS["bleed"]
            factors.append(
                f"Lesion bleeds "
                f"(+{RISK_WEIGHTS['bleed']} pts)")

        if self._encode_symptom(symptoms.get("grew")) == 1.0:
            score += RISK_WEIGHTS["grew"]
            factors.append(
                f"Lesion grew "
                f"(+{RISK_WEIGHTS['grew']} pts)")

        if self._encode_symptom(symptoms.get("elevation")) == 1.0:
            score += RISK_WEIGHTS["elevation"]
            factors.append(
                f"Lesion is raised "
                f"(+{RISK_WEIGHTS['elevation']} pts)")

        if self._encode_symptom(symptoms.get("hurt")) == 1.0:
            score += RISK_WEIGHTS["hurt"]
            factors.append(
                f"Lesion hurts "
                f"(+{RISK_WEIGHTS['hurt']} pts)")

        if self._encode_symptom(symptoms.get("itch")) == 1.0:
            score += RISK_WEIGHTS["itch"]
            factors.append(
                f"Lesion itches "
                f"(+{RISK_WEIGHTS['itch']} pts)")

        # Patient history
        if self._encode_bool(symptoms.get("skin_cancer_history")):
            score += RISK_WEIGHTS["skin_cancer_history"]
            factors.append(
                f"Previous skin cancer "
                f"(+{RISK_WEIGHTS['skin_cancer_history']} pts)")

        if self._encode_bool(symptoms.get("cancer_history")):
            score += RISK_WEIGHTS["cancer_history"]
            factors.append(
                f"Previous cancer "
                f"(+{RISK_WEIGHTS['cancer_history']} pts)")

        if self._encode_bool(symptoms.get("smoke")):
            score += RISK_WEIGHTS["smoke"]
            factors.append(
                f"Smoker "
                f"(+{RISK_WEIGHTS['smoke']} pts)")

        if self._encode_bool(symptoms.get("drink")):
            score += RISK_WEIGHTS["drink"]
            factors.append(
                f"Drinker "
                f"(+{RISK_WEIGHTS['drink']} pts)")

        # Age — validated inside scoring
        age_raw = symptoms.get("age")
        if age_raw is not None:
            try:
                age = int(float(age_raw))
                age = max(1, min(120, age))
                if age >= 70:
                    score += RISK_WEIGHTS["age_over_70"]
                    factors.append(
                        f"Age {age} (over 70) "
                        f"(+{RISK_WEIGHTS['age_over_70']} pts)")
                elif age >= 60:
                    score += RISK_WEIGHTS["age_60_to_70"]
                    factors.append(
                        f"Age {age} (60-70) "
                        f"(+{RISK_WEIGHTS['age_60_to_70']} pts)")
                elif age >= 50:
                    score += RISK_WEIGHTS["age_50_to_60"]
                    factors.append(
                        f"Age {age} (50-60) "
                        f"(+{RISK_WEIGHTS['age_50_to_60']} pts)")
            except (TypeError, ValueError):
                pass

        # Lesion size
        d1_raw = symptoms.get("diameter_1")
        if d1_raw is not None:
            try:
                d1 = float(d1_raw)
                d1 = max(0.0, d1)
                if d1 > 15:
                    score += RISK_WEIGHTS["diameter_over_15"]
                    factors.append(
                        f"Lesion size {d1:.0f}mm (>15mm) "
                        f"(+{RISK_WEIGHTS['diameter_over_15']} pts)")
                elif d1 >= 10:
                    score += RISK_WEIGHTS["diameter_10_to_15"]
                    factors.append(
                        f"Lesion size {d1:.0f}mm (10-15mm) "
                        f"(+{RISK_WEIGHTS['diameter_10_to_15']} pts)")
            except (TypeError, ValueError):
                pass

        # Skin tone
        fitz_raw = symptoms.get("fitspatrick")
        if fitz_raw is not None:
            try:
                fitz = float(fitz_raw)
                fitz = max(1.0, min(6.0, fitz))
                if fitz >= 5:
                    score += RISK_WEIGHTS["dark_skin"]
                    factors.append(
                        f"Dark skin tone (FST {int(fitz)}) "
                        f"(+{RISK_WEIGHTS['dark_skin']} pts)")
            except (TypeError, ValueError):
                pass

        return min(score, 100), factors

    def _get_risk_level(self, score: int) -> str:
        """Map score to risk level string."""
        for level, (low, high) in RISK_LEVELS.items():
            if low <= score < high:
                return level
        return "CRITICAL"

    # ─── ML Model Probability ──────────────────────────────────────

    def _ml_probability(self, symptoms: dict) -> float:
        """
        Get RandomForest malignancy probability.
        Returns 0.0-1.0 or -1.0 if model not loaded.

        FIX 1: Diameter explicit None check.
          0 is a valid lesion diameter. Code such as:
            float(symptoms.get("diameter_1") or median)
          would replace 0 with median because 0 is falsy.
          Explicit None check prevents this silent error.

        FIX 2: Age clipped to 1-100.
        FIX 3: Fitzpatrick clipped to 1-6.
        FIX 4: Full try/except — bad input never crashes agent.
        """
        if self.model is None:
            return -1.0

        try:
            d = self.model_data

            # Age — validate and clip
            age_raw = symptoms.get("age", 60)
            try:
                age = float(age_raw)
                age = max(1.0, min(100.0, age))
            except (TypeError, ValueError):
                age = 60.0

            # Fitzpatrick — validate and clip to 1-6
            fitz_raw = symptoms.get("fitspatrick")
            if fitz_raw is not None:
                try:
                    fitz = float(fitz_raw)
                    fitz = max(1.0, min(6.0, fitz))
                except (TypeError, ValueError):
                    fitz = float(d["fitz_median"])
            else:
                fitz = float(d["fitz_median"])

            # FIX 1: diameter_1 — explicit None check
            # 0 is a valid diameter. Must use None check, not
            # (val or median) which treats 0 as missing.
            d1_raw = symptoms.get("diameter_1")
            if d1_raw is not None:
                try:
                    d1 = float(d1_raw)
                    d1 = max(0.0, d1)
                except (TypeError, ValueError):
                    d1 = float(d["d1_median"])
            else:
                d1 = float(d["d1_median"])

            # FIX 1: diameter_2 — same explicit None check
            d2_raw = symptoms.get("diameter_2")
            if d2_raw is not None:
                try:
                    d2 = float(d2_raw)
                    d2 = max(0.0, d2)
                except (TypeError, ValueError):
                    d2 = float(d["d2_median"])
            else:
                d2 = float(d["d2_median"])

            # Gender
            gender_raw = symptoms.get("gender")
            if gender_raw is None:
                gender = 0.5
            else:
                gender = 1.0 if str(
                    gender_raw).strip().upper() == "MALE" else 0.0

            # Feature row — order must match ALL_FEATURES in config.py
            # age, itch, grew, hurt, changed, bleed, elevation,
            # skin_cancer_history, cancer_history, fitspatrick,
            # gender, diameter_1, diameter_2, smoke, drink
            row = [
                age,
                self._encode_symptom(symptoms.get("itch")),
                self._encode_symptom(symptoms.get("grew")),
                self._encode_symptom(symptoms.get("hurt")),
                self._encode_symptom(symptoms.get("changed")),
                self._encode_symptom(symptoms.get("bleed")),
                self._encode_symptom(symptoms.get("elevation")),
                self._encode_bool(symptoms.get("skin_cancer_history")),
                self._encode_bool(symptoms.get("cancer_history")),
                fitz,
                gender,
                d1,
                d2,
                self._encode_bool(symptoms.get("smoke")),
                self._encode_bool(symptoms.get("drink")),
            ]

            X    = np.array(row, dtype=np.float32).reshape(1, -1)
            prob = self.model.predict_proba(X)[0][1]
            return float(prob)

        except Exception as e:
            print(f"  [Clinical] ML prediction error: {e}")
            traceback.print_exc()
            return -1.0

    # ─── Report Generation ─────────────────────────────────────────

    def _generate_report(
        self,
        symptoms    : dict,
        rule_score  : int,
        risk_level  : str,
        risk_factors: list,
        ml_prob     : float,
        case_id     : str,
    ) -> str:
        """Generate written clinical risk report."""
        rec = RISK_RECOMMENDATIONS[risk_level]

        ml_line = (
            f"  ML model probability : {ml_prob*100:.1f}%"
            if ml_prob >= 0 else
            "  ML model probability : Not available"
        )

        if risk_factors:
            factors_text = "\n".join(
                f"  \u2717 {f}" for f in risk_factors)
        else:
            factors_text = (
                "  \u2713 No significant risk factors identified")

        return (
            f"MAAI-SkinDx | Clinical Agent \u2014 Risk Assessment\n"
            f"Case ID   : {case_id}\n"
            f"{'='*60}\n\n"
            f"PATIENT INFORMATION\n"
            f"  Age              : "
            f"{symptoms.get('age', 'Not provided')}\n"
            f"  Gender           : "
            f"{symptoms.get('gender', 'Not provided')}\n"
            f"  Skin tone (FST)  : "
            f"{symptoms.get('fitspatrick', 'Not provided')}\n"
            f"  Region           : "
            f"{symptoms.get('region', 'Not provided')}\n\n"
            f"LESION SYMPTOMS\n"
            f"  Changed recently : "
            f"{symptoms.get('changed', 'Not provided')}\n"
            f"  Grew             : "
            f"{symptoms.get('grew', 'Not provided')}\n"
            f"  Bleeds           : "
            f"{symptoms.get('bleed', 'Not provided')}\n"
            f"  Elevated / Raised: "
            f"{symptoms.get('elevation', 'Not provided')}\n"
            f"  Hurts            : "
            f"{symptoms.get('hurt', 'Not provided')}\n"
            f"  Itches           : "
            f"{symptoms.get('itch', 'Not provided')}\n\n"
            f"PATIENT HISTORY\n"
            f"  Skin cancer history : "
            f"{symptoms.get('skin_cancer_history', 'Not provided')}\n"
            f"  Cancer history      : "
            f"{symptoms.get('cancer_history', 'Not provided')}\n"
            f"  Smoker              : "
            f"{symptoms.get('smoke', 'Not provided')}\n\n"
            f"RISK ASSESSMENT\n"
            f"  Rule-based score : {rule_score} / 100\n"
            f"{ml_line}\n"
            f"  Risk level       : {risk_level}\n\n"
            f"RISK FACTORS IDENTIFIED\n"
            f"{factors_text}\n\n"
            f"RECOMMENDATION\n"
            f"  {rec}\n\n"
            f"IMPORTANT NOTICE\n"
            f"  This clinical risk assessment is based on patient-\n"
            f"  reported symptoms and should be interpreted alongside\n"
            f"  the AI image diagnosis (Agent 3) and visual\n"
            f"  explanation (Agent 4). Final clinical decisions must\n"
            f"  be made by a qualified clinician.\n\n"
            f"{'='*60}\n"
        )

    # ─── Main Entry Point ──────────────────────────────────────────

    def run(self, symptoms: dict, case_id: str) -> dict:
        """
        Assess clinical risk from patient symptom form.
        Called by Orchestrator in parallel with Agent 4.

        Args:
            symptoms : patient symptom dict from form
            case_id  : unique case identifier

        Returns dict with status, risk_score, risk_level,
        risk_factors, ml_probability, recommendation,
        report, report_path
        """
        print(f"\n{'='*60}")
        print(f"  {self.name}")
        print(f"  Case : {case_id}")
        print(f"{'='*60}")

        if not symptoms:
            return {
                "status"        : "SKIP",
                "reason"        : "No symptom data provided.",
                "risk_score"    : None,
                "risk_level"    : None,
                "risk_factors"  : [],
                "ml_probability": None,
                "recommendation": None,
                "report"        : None,
                "report_path"   : None,
            }

        # Rule-based scoring
        print(f"\n  -- Rule-based Scoring --")
        rule_score, risk_factors = self._rule_based_score(symptoms)
        risk_level = self._get_risk_level(rule_score)
        print(f"  Rule score  : {rule_score} / 100")
        print(f"  Risk level  : {risk_level}")
        print(f"  Risk factors: {len(risk_factors)}")
        for f in risk_factors:
            print(f"    \u2717 {f}")

        # ML probability
        print(f"\n  -- ML Model Probability --")
        ml_prob = self._ml_probability(symptoms)
        if ml_prob >= 0:
            print(f"  ML prob : {ml_prob*100:.1f}%")
        else:
            print(f"  ML prob : Not available")

        recommendation = RISK_RECOMMENDATIONS[risk_level]
        print(f"\n  Recommendation : {recommendation}")

        # Generate and save report
        report = self._generate_report(
            symptoms, rule_score, risk_level,
            risk_factors, ml_prob, case_id)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        report_path = os.path.join(
            OUTPUT_DIR, f"{case_id}_clinical_report.txt")

        # UTF-8 required — report contains unicode symbols
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"  Report saved : {report_path}")
        print(f"{'='*60}\n")

        return {
            "status"        : "PASS",
            "reason"        : "Clinical risk assessment complete.",
            "risk_score"    : rule_score,
            "risk_level"    : risk_level,
            "risk_factors"  : risk_factors,
            "ml_probability": round(ml_prob, 4) if ml_prob >= 0
                              else None,
            "recommendation": recommendation,
            "report"        : report,
            "report_path"   : report_path,
        }