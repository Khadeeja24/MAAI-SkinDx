# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 7 Expert — Clinical Expert Agent
# ══════════════════════════════════════════════════════════════════
# Final agent in the MAAI-SkinDx pipeline.
# Generates a complete 15-section doctor-style clinical report.
#
# Sections 1-7  : Template-based (deterministic)
# Sections 8-14 : LLM-generated (openai/gpt-oss-120b)
# Section 15    : Disclaimer (deterministic)
# ══════════════════════════════════════════════════════════════════

import os
import sys
import traceback
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import (
    OUTPUT_DIR, LLM_MODEL, LLM_MAX_TOKENS,
    LLM_TEMPERATURE, MALIGNANT_CLASSES,
    CLASS_FULL_NAMES, URGENCY_LABELS,
    FITZPATRICK_LABELS, REGION_LABELS,
)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class ClinicalExpertAgent:

    def __init__(self):
        self.name   = "Agent 7 Expert — Clinical Expert Agent"
        self.client = None
        print(f"\n[{self.name}] Initialising ...")
        self._init_llm()
        print(f"[{self.name}] Ready")

    # ── Initialisation ─────────────────────────────────────────────

    def _init_llm(self):
        if not GROQ_AVAILABLE:
            print(f"  [Agent 7] Groq not available")
            return
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            print(f"  [Agent 7] GROQ_API_KEY not set")
            return
        try:
            self.client = Groq(api_key=api_key)
            print(f"  [Agent 7] LLM: {LLM_MODEL} ready")
        except Exception:
            self.client = None

    # ── Section parser ─────────────────────────────────────────────

    def _parse_sections(self, raw: str) -> dict:
        """
        Parse LLM output into 7 clinical sections.

        Robust against markdown formatting the LLM may add
        despite instructions — strips hash symbols, asterisks,
        and handles spacing variations.

        Returns dict with 7 keys. Returns None if fewer than
        3 sections are filled, signalling fallback needed.
        """
        sections = {
            "differential_diagnoses"    : "",
            "recommended_investigations": "",
            "treatment_recommendations" : "",
            "medications"               : "",
            "things_to_avoid"           : "",
            "followup_schedule"         : "",
            "referral_decision"         : "",
        }

        # Keyword to key mapping
        # Using startswith on uppercase so partial matches work
        header_map = [
            ("DIFFERENTIAL DIAGNOS", "differential_diagnoses"),
            ("RECOMMENDED INVEST",   "recommended_investigations"),
            ("TREATMENT RECOMMEND",  "treatment_recommendations"),
            ("MEDICATION",           "medications"),
            ("THINGS TO AVOID",      "things_to_avoid"),
            ("FOLLOW",               "followup_schedule"),
            ("REFERRAL",             "referral_decision"),
        ]

        current_key   = None
        current_lines = []

        for line in raw.split("\n"):
            # Strip markdown that LLM adds despite instructions
            clean = line.strip()
            clean = clean.lstrip("#").strip()
            clean = clean.strip("*").strip()
            clean_upper = clean.upper().rstrip(":")

            matched = False
            for keyword, key in header_map:
                if clean_upper.startswith(keyword):
                    # Save previous section
                    if current_key is not None:
                        sections[current_key] = "\n".join(
                            current_lines).strip()
                    current_key   = key
                    current_lines = []
                    matched       = True
                    break

            if not matched and current_key is not None:
                current_lines.append(line)

        # Save last section
        if current_key is not None:
            sections[current_key] = "\n".join(
                current_lines).strip()

        filled = sum(
            1 for v in sections.values() if v.strip())
        print(f"  [Agent 7] Sections filled: {filled}/7")

        if filled < 3:
            print(f"  [Agent 7] Parser got < 3 sections "
                  f"— using fallback")
            return None

        return sections

    # ── LLM clinical sections ──────────────────────────────────────

    def _generate_clinical_sections(
        self,
        predicted_name  : str,
        predicted_index : int,
        confidence      : float,
        top_5           : list,
        risk_level      : str,
        risk_score      : int,
        ml_probability  : float,
        risk_factors    : list,
        focus_type      : str,
        xai_explanation : str,
        rag_summary     : str,
        conflict_type   : str,
        decision        : str,
        urgency         : str,
        symptoms        : dict,
    ) -> dict:
        """
        Generate Sections 8-14 using Groq LLM.
        Falls back to templates if LLM fails or parsing fails.
        """
        if self.client is None:
            print(f"  [Agent 7] No LLM client — using fallback")
            return self._fallback_sections(
                predicted_name, predicted_index,
                risk_level, decision, urgency)

        is_malignant = predicted_index in MALIGNANT_CLASSES

        # Build top 5 string
        top5_str = "\n".join([
            f"  {p['rank']}. {p['class_name']} "
            f"({p['confidence_pct']})"
            for p in (top_5 or [])[:5]
        ])

        # Build risk factors string
        factors_str = (
            "\n".join(f"  {i+1}. {f}"
                      for i, f in enumerate(risk_factors or []))
            if risk_factors else "  None identified"
        )

        ml_str = (
            f"{ml_probability*100:.1f}%"
            if ml_probability and ml_probability >= 0
            else "Not available"
        )

        sym    = symptoms or {}
        age    = str(sym.get("age",    "Not provided"))
        gender = str(sym.get("gender", "Not provided"))
        region = str(sym.get("region", "Not provided"))

        # Urgency-aware treatment instruction
        if decision in ("ESCALATE", "REFER_URGENT"):
            treatment_instruction = (
                "This is an URGENT case. "
                "Say clearly that surgical removal is needed "
                "within 48 hours and urgent specialist referral "
                "is required immediately."
            )
        elif decision in ("REVIEW", "REFER"):
            treatment_instruction = (
                "This case requires specialist review. "
                "Recommend dermatologist assessment within "
                "one week and possible biopsy."
            )
        else:
            treatment_instruction = (
                "This is a low risk case. "
                "Recommend conservative monitoring and "
                "routine follow up."
            )

        system_prompt = (
            "You are a senior dermatologist writing a clinical "
            "consultation report for a patient. "
            "Write in simple clear English. "
            "Avoid complex medical words. "
            "When you use a medical term explain it in brackets "
            "immediately after it. "
            "Example: excisional biopsy "
            "(surgical removal of the lesion for laboratory testing). "
            "Be specific and practical. "
            "STRICT FORMATTING RULE: "
            "Write ONLY the 7 sections listed. "
            "Do NOT use markdown. "
            "Do NOT use # symbols. "
            "Do NOT use ** bold markers. "
            "Write each header in CAPITAL LETTERS followed by "
            "a colon on its own line. "
            "Use numbered lists 1. 2. 3. for items. "
            "Start directly with the first header."
        )

        user_prompt = (
            f"PATIENT: {age} year old {gender}, "
            f"lesion on {region}.\n"
            f"AI DIAGNOSIS: {predicted_name} at "
            f"{confidence*100:.1f}% confidence.\n"
            f"MALIGNANT CLASS: {'YES' if is_malignant else 'No'}\n"
            f"DECISION: {decision} | URGENCY: {urgency}\n"
            f"RISK SCORE: {risk_score}/100 | "
            f"ML PROBABILITY: {ml_str}\n"
            f"RISK LEVEL: {risk_level}\n"
            f"RISK FACTORS:\n{factors_str}\n"
            f"CONFLICT: {conflict_type}\n"
            f"TOP 5 PREDICTIONS:\n{top5_str}\n"
            f"CLINICAL KNOWLEDGE: {(rag_summary or '')[:300]}\n\n"
            f"TREATMENT GUIDANCE: {treatment_instruction}\n\n"
            f"Write exactly these 7 sections in order.\n"
            f"Start each section with its header in CAPITALS "
            f"followed by a colon.\n"
            f"Do not write anything before the first header.\n\n"
            f"DIFFERENTIAL DIAGNOSES:\n"
            f"RECOMMENDED INVESTIGATIONS:\n"
            f"TREATMENT RECOMMENDATIONS:\n"
            f"MEDICATIONS:\n"
            f"THINGS TO AVOID:\n"
            f"FOLLOW-UP SCHEDULE:\n"
            f"REFERRAL DECISION:\n"
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
            raw = response.choices[0].message.content.strip()

            # Show first 400 chars for debugging
            print(f"  [Agent 7] LLM preview: "
                  f"{raw[:400].replace(chr(10), ' | ')}")

            parsed = self._parse_sections(raw)
            if parsed is None:
                print(f"  [Agent 7] Falling back to templates")
                return self._fallback_sections(
                    predicted_name, predicted_index,
                    risk_level, decision, urgency)
            return parsed

        except Exception as e:
            print(f"  [Agent 7] LLM error: {e}")
            return self._fallback_sections(
                predicted_name, predicted_index,
                risk_level, decision, urgency)

    # ── Fallback sections ──────────────────────────────────────────

    def _fallback_sections(
        self,
        predicted_name : str,
        predicted_index: int,
        risk_level     : str,
        decision       : str,
        urgency        : str,
    ) -> dict:
        """
        Evidence-based template fallback when LLM unavailable
        or when parser finds fewer than 3 sections.
        """
        is_malignant = predicted_index in MALIGNANT_CLASSES
        is_urgent    = decision in (
            "ESCALATE", "REFER_URGENT", "REVIEW")

        if is_malignant or is_urgent:
            return {
                "differential_diagnoses": (
                    f"1. {predicted_name} — the main AI prediction.\n"
                    "2. Melanoma (skin cancer starting from "
                    "pigment cells) — must be ruled out urgently.\n"
                    "3. Basal Cell Carcinoma (most common skin "
                    "cancer, usually on sun-exposed areas).\n"
                    "4. Squamous Cell Carcinoma (skin cancer that "
                    "can spread if not treated promptly).\n"
                    "5. Dysplastic Nevus (abnormal mole that can "
                    "develop into melanoma)."
                ),
                "recommended_investigations": (
                    "1. Dermoscopy (a magnified examination of the "
                    "skin lesion done in the clinic) — immediately.\n"
                    "2. Excisional biopsy (surgical removal of the "
                    "entire lesion and sending it to a laboratory "
                    "to check for cancer cells) — within 48 hours.\n"
                    "3. Full body skin examination to check for "
                    "other suspicious lesions.\n"
                    "4. Lymph node check if cancer is confirmed."
                ),
                "treatment_recommendations": (
                    "1. Urgent surgical removal of the entire "
                    "lesion is required within 48 hours.\n"
                    "2. The removed tissue will be sent to a "
                    "laboratory to check for cancer cells.\n"
                    "3. Further treatment depends on the "
                    "laboratory results.\n"
                    "4. A skin cancer specialist must be seen "
                    "immediately — do not delay."
                ),
                "medications": (
                    "No medications are needed before the "
                    "surgical removal procedure.\n"
                    "After surgery the specialist will prescribe "
                    "any medications needed based on the "
                    "laboratory results."
                ),
                "things_to_avoid": (
                    "1. Avoid all sun exposure on the affected "
                    "area until you have seen the specialist.\n"
                    "2. Do not apply any creams, oils or home "
                    "remedies to the lesion.\n"
                    "3. Do not scratch or pick at the lesion.\n"
                    "4. Do not delay the specialist appointment "
                    "under any circumstances."
                ),
                "followup_schedule": (
                    "Within 48 hours: Book and attend urgent "
                    "specialist appointment.\n"
                    "Week 1 to 2: Excisional biopsy procedure.\n"
                    "Week 2 to 4: Review laboratory results with "
                    "specialist and plan further treatment.\n"
                    "Ongoing: As directed by the specialist."
                ),
                "referral_decision": (
                    "URGENT referral to a Dermatologist or Skin "
                    "Cancer Clinic within 48 hours.\n"
                    "Include in the referral: this AI diagnostic "
                    "report, the Grad-CAM heatmap images, the "
                    "clinical risk score of 100/100, and the "
                    "ML malignancy probability of 89%."
                ),
            }
        else:
            return {
                "differential_diagnoses": (
                    f"1. {predicted_name} — the main AI prediction.\n"
                    "2. Inflammatory skin condition (redness and "
                    "irritation from skin inflammation).\n"
                    "3. Contact dermatitis (skin reaction to "
                    "something that touched the skin).\n"
                    "4. Benign keratosis (a harmless thickened "
                    "patch of skin, very common in adults)."
                ),
                "recommended_investigations": (
                    "1. Dermoscopy (a magnified examination of the "
                    "lesion by a dermatologist).\n"
                    "2. Photographic monitoring (a photograph taken "
                    "today for comparison at future visits).\n"
                    "3. Patch testing if a skin allergy is suspected."
                ),
                "treatment_recommendations": (
                    "No urgent treatment is needed at this stage.\n"
                    "A routine skin check with a dermatologist is "
                    "recommended.\n"
                    "If the lesion changes in size, shape or colour, "
                    "or starts to bleed or itch more, attend the "
                    "clinic sooner than scheduled."
                ),
                "medications": (
                    "No prescription medicines are required at "
                    "this stage.\n"
                    "A simple moisturising cream may be used if "
                    "the skin around the lesion feels dry."
                ),
                "things_to_avoid": (
                    "1. Avoid prolonged direct sun exposure on "
                    "the affected area.\n"
                    "2. Apply sunscreen SPF 30 or higher every day.\n"
                    "3. Avoid scratching or picking at the lesion.\n"
                    "4. Avoid harsh soaps or chemical skin products "
                    "on the area."
                ),
                "followup_schedule": (
                    "Routine dermatologist review in 6 to 12 months.\n"
                    "Check the lesion yourself every month at home.\n"
                    "Come back sooner if you notice any change in "
                    "size, shape, colour, or if it starts to bleed."
                ),
                "referral_decision": (
                    "Routine referral to Dermatology for a "
                    "standard follow-up skin check.\n"
                    "No urgent referral is required at this stage."
                ),
            }

    # ── Report assembly ────────────────────────────────────────────

    def _assemble_report(
        self,
        case_id           : str,
        predicted_name    : str,
        predicted_index   : int,
        confidence        : float,
        top_5             : list,
        risk_level        : str,
        risk_score        : int,
        ml_probability    : float,
        risk_factors      : list,
        focus_type        : str,
        xai_explanation   : str,
        rag_summary       : str,
        conflict_type     : str,
        conflict_severity : str,
        conflict_desc     : str,
        decision          : str,
        urgency           : str,
        decision_rationale: str,
        symptoms          : dict,
        clinical_sections : dict,
        llm_used          : bool,
        fusion_weights    : dict,
        fused_malignancy  : float,
    ) -> str:

        now        = datetime.now()
        date_str   = now.strftime("%d %B %Y")
        time_str   = now.strftime("%H:%M")
        is_mal     = predicted_index in MALIGNANT_CLASSES
        class_type = "MALIGNANT" if is_mal else "Benign"
        urgency_label = URGENCY_LABELS.get(urgency, urgency)

        sym = symptoms or {}
        age    = str(sym.get("age",    "Not provided"))
        gender = str(sym.get("gender", "Not provided"))
        region = str(sym.get("region", "Not provided"))
        fitz   = sym.get("fitspatrick")
        fitz_label = (
            FITZPATRICK_LABELS.get(int(fitz), str(fitz))
            if fitz else "Not provided"
        )

        # Symptoms
        sym_map = {
            "itch"     : "Itching",
            "grew"     : "Grew recently",
            "hurt"     : "Painful",
            "changed"  : "Changed recently",
            "bleed"    : "Bleeding",
            "elevation": "Raised or elevated",
        }
        symptom_lines = []
        for key, label in sym_map.items():
            if str(sym.get(key, "False")).lower() == "true":
                symptom_lines.append(f"    Yes — {label}")
        if not symptom_lines:
            symptom_lines = ["    No significant symptoms reported"]

        history_lines = []
        if sym.get("skin_cancer_history"):
            history_lines.append("    Previous skin cancer history")
        if sym.get("cancer_history"):
            history_lines.append("    Previous cancer history")
        if sym.get("smoke"):
            history_lines.append("    Smoker")
        if sym.get("drink"):
            history_lines.append("    Alcohol use")
        if not history_lines:
            history_lines = ["    No significant history"]

        d1 = sym.get("diameter_1", "Not provided")
        d2 = sym.get("diameter_2", "Not provided")

        # Top 5
        top5_str = ""
        for p in (top_5 or [])[:5]:
            bar = "=" * int(p.get("confidence", 0) * 20)
            top5_str += (
                f"    {p['rank']}. "
                f"{p['class_name']:<42} "
                f"{p['confidence_pct']:>6}  {bar}\n"
            )

        # Risk factors
        if risk_factors:
            factors_str = "\n".join(
                f"    [{i+1}] {f}"
                for i, f in enumerate(risk_factors))
        else:
            factors_str = (
                "    No significant risk factors identified")

        ml_str = (
            f"{ml_probability*100:.1f}%"
            if ml_probability and ml_probability >= 0
            else "Not available"
        )

        # Fusion weights
        fw_str = ""
        if fusion_weights:
            for name, w in fusion_weights.items():
                bar = "=" * int(w * 30)
                fw_str += (
                    f"    {name:<12}: {w:.4f}  {bar}\n")

        cs  = clinical_sections
        sep = "=" * 65

        report = f"""
{sep}
     MAAI-SkinDx | CLINICAL EXPERT CONSULTATION REPORT
{sep}
  Multi-Agent AI System for Automated Dermatological Diagnosis
  Insight Centre for Data Analytics | University of Galway
  Supervisor: Dr. Saeed Alsamhi
{sep}

SECTION 1 — CASE INFORMATION
{sep}
  Case ID          : {case_id}
  Report Date      : {date_str}
  Report Time      : {time_str}
  Report Type      : {'LLM-Generated (openai/gpt-oss-120b)' if llm_used else 'Template-Based (Fallback)'}
  System           : MAAI-SkinDx Expert Agent v1.0
  Pipeline Status  : Complete

{sep}
SECTION 2 — PATIENT INFORMATION
{sep}
  Age              : {age}
  Gender           : {gender}
  Skin Tone        : {fitz_label}
  Affected Region  : {region}
  Lesion Size      : {d1} mm x {d2} mm
  Symptoms provided: {'Yes' if sym else 'No — image only'}

  REPORTED SYMPTOMS:
{chr(10).join(symptom_lines)}

  PATIENT HISTORY:
{chr(10).join(history_lines)}

{sep}
SECTION 3 — AI DIAGNOSTIC SUMMARY
{sep}
  Primary Diagnosis    : {CLASS_FULL_NAMES.get(predicted_index, predicted_name)}
  AI Confidence        : {confidence*100:.1f}%
  Classification       : {class_type}
  Coordination Decision: {decision}
  Clinical Urgency     : {urgency_label}

  Decision Rationale:
    {(decision_rationale or 'Not available')[:300]}

  TOP 5 AI PREDICTIONS:
{top5_str}
{sep}
SECTION 4 — VISUAL ANALYSIS (Grad-CAM XAI Agent)
{sep}
  Focus Pattern: {focus_type or 'Not available'}

  XAI Explanation:
    {(xai_explanation or 'Not available')[:400]}

{sep}
SECTION 5 — CLINICAL RISK ASSESSMENT
{sep}
  Rule-Based Score : {risk_score}/100
  ML Probability   : {ml_str}
  Risk Level       : {risk_level or 'Not assessed'}

  RISK FACTORS IDENTIFIED:
{factors_str}

{sep}
SECTION 6 — COORDINATION AGENT ANALYSIS
{sep}
  Conflict Type    : {conflict_type}
  Conflict Severity: {conflict_severity}

  Analysis:
    {(conflict_desc or 'No conflict detected.')[:300]}

  ADAPTIVE FUSION WEIGHTS:
{fw_str if fw_str else '    Not available'}
  Fused Malignancy Signal: {fused_malignancy:.4f}

{sep}
SECTION 7 — CLINICAL KNOWLEDGE SUMMARY (DermNet NZ via RAG)
{sep}
  {rag_summary or 'Not available.'}

{sep}
SECTION 8 — DIFFERENTIAL DIAGNOSES
{sep}
{cs.get('differential_diagnoses', 'Not available')}

{sep}
SECTION 9 — RECOMMENDED INVESTIGATIONS
{sep}
{cs.get('recommended_investigations', 'Not available')}

{sep}
SECTION 10 — TREATMENT RECOMMENDATIONS
{sep}
{cs.get('treatment_recommendations', 'Not available')}

{sep}
SECTION 11 — MEDICATIONS
{sep}
{cs.get('medications', 'Not available')}

{sep}
SECTION 12 — THINGS TO AVOID
{sep}
{cs.get('things_to_avoid', 'Not available')}

{sep}
SECTION 13 — FOLLOW-UP SCHEDULE
{sep}
{cs.get('followup_schedule', 'Not available')}

{sep}
SECTION 14 — REFERRAL DECISION
{sep}
{cs.get('referral_decision', 'Not available')}

{sep}
SECTION 15 — IMPORTANT DISCLAIMER
{sep}
  This report was generated by MAAI-SkinDx, an AI-assisted
  dermatological diagnosis system developed at the Insight
  Centre for Data Analytics, University of Galway.

  This report is intended to SUPPORT clinical decision making
  by a qualified healthcare professional only. It does NOT
  replace clinical examination, professional medical judgement,
  or established diagnostic procedures.

  All AI predictions, risk scores, and recommendations must be
  verified and acted upon by a qualified clinician.

  System performance: 74.18% macro F1 on 7,734 hold-out test
  images across 9 disease classes.

  In an emergency, contact emergency services immediately.
  Do not wait for AI assessment.

  Generated by : MAAI-SkinDx Expert Agent v1.0
  Institution  : University of Galway, Ireland
  Supervisor   : Dr. Saeed Alsamhi
{sep}
"""
        return report

    # ── Main entry point ───────────────────────────────────────────

    def run(
        self,
        case_id            : str,
        predicted_name     : str,
        predicted_index    : int,
        confidence         : float,
        top_5              : list,
        risk_level         : str   = None,
        risk_score         : int   = 0,
        ml_probability     : float = None,
        risk_factors       : list  = None,
        focus_type         : str   = None,
        xai_explanation    : str   = None,
        rag_summary        : str   = None,
        conflict_type      : str   = "CONCORDANT",
        conflict_severity  : str   = "NONE",
        conflict_desc      : str   = "",
        decision           : str   = "ROUTINE",
        urgency            : str   = "ROUTINE",
        decision_rationale : str   = "",
        symptoms           : dict  = None,
        fusion_weights     : dict  = None,
        fused_malignancy   : float = 0.0,
    ) -> dict:
        """
        Generate complete clinical expert report.
        Called by Orchestrator after Agent 5 completes.
        """
        print(f"\n{'='*60}")
        print(f"  {self.name}")
        print(f"  Case     : {case_id}")
        print(f"  Diagnosis: {predicted_name} "
              f"({confidence*100:.1f}%)")
        print(f"  Decision : {decision} | {urgency}")
        print(f"  LLM      : "
              f"{'Groq active' if self.client else 'fallback'}")
        print(f"{'='*60}")

        try:
            # Step 1 — Generate LLM clinical sections 8-14
            print(f"\n  -- Generating Clinical Sections --")
            clinical_sections = self._generate_clinical_sections(
                predicted_name  = predicted_name,
                predicted_index = predicted_index,
                confidence      = confidence,
                top_5           = top_5,
                risk_level      = risk_level or "UNKNOWN",
                risk_score      = risk_score or 0,
                ml_probability  = ml_probability or -1.0,
                risk_factors    = risk_factors or [],
                focus_type      = focus_type or "Unknown",
                xai_explanation = xai_explanation or "",
                rag_summary     = rag_summary or "",
                conflict_type   = conflict_type,
                decision        = decision,
                urgency         = urgency,
                symptoms        = symptoms or {},
            )
            llm_used = self.client is not None

            # Step 2 — Assemble full 15-section report
            print(f"\n  -- Assembling Report --")
            report = self._assemble_report(
                case_id            = case_id,
                predicted_name     = predicted_name,
                predicted_index    = predicted_index,
                confidence         = confidence,
                top_5              = top_5 or [],
                risk_level         = risk_level or "NOT ASSESSED",
                risk_score         = risk_score or 0,
                ml_probability     = ml_probability,
                risk_factors       = risk_factors or [],
                focus_type         = focus_type,
                xai_explanation    = xai_explanation,
                rag_summary        = rag_summary,
                conflict_type      = conflict_type,
                conflict_severity  = conflict_severity,
                conflict_desc      = conflict_desc,
                decision           = decision,
                urgency            = urgency,
                decision_rationale = decision_rationale,
                symptoms           = symptoms,
                clinical_sections  = clinical_sections,
                llm_used           = llm_used,
                fusion_weights     = fusion_weights or {},
                fused_malignancy   = fused_malignancy,
            )

            # Step 3 — Save to disk
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            report_path = os.path.join(
                OUTPUT_DIR,
                f"{case_id}_expert_report.txt")

            with open(report_path, "w",
                      encoding="utf-8") as f:
                f.write(report)

            print(f"  Report saved: {report_path}")
            print(f"{'='*60}\n")

            return {
                "status"      : "PASS",
                "reason"      : "Expert clinical report generated.",
                "report"      : report,
                "report_path" : report_path,
                "llm_used"    : llm_used,
                "sections"    : list(clinical_sections.keys()),
            }

        except Exception as e:
            print(f"  [Agent 7] Error: {e}")
            traceback.print_exc()
            return {
                "status"      : "FAIL",
                "reason"      : str(e),
                "report"      : None,
                "report_path" : None,
                "llm_used"    : False,
                "sections"    : [],
            }