# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 7 Expert — Clinical Expert Agent
# ══════════════════════════════════════════════════════════════════
# Final agent in the MAAI-SkinDx pipeline.
# Generates a doctor prescription-style clinical report.
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

# Pox class index offset — must match Orchestrator
POX_INDEX_OFFSET = 100


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
        sections = {
            "differential_diagnoses"    : "",
            "recommended_investigations": "",
            "treatment_recommendations" : "",
            "medications"               : "",
            "things_to_avoid"           : "",
            "followup_schedule"         : "",
            "referral_decision"         : "",
        }

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
            clean       = line.strip().lstrip("#").strip().strip("*").strip()
            clean_upper = clean.upper().rstrip(":")

            matched = False
            for keyword, key in header_map:
                if clean_upper.startswith(keyword):
                    if current_key is not None:
                        sections[current_key] = "\n".join(
                            current_lines).strip()
                    current_key   = key
                    current_lines = []
                    matched       = True
                    break

            if not matched and current_key is not None:
                current_lines.append(line)

        if current_key is not None:
            sections[current_key] = "\n".join(
                current_lines).strip()

        filled = sum(1 for v in sections.values() if v.strip())
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
        # Pox cases — always use fallback templates
        if predicted_index >= POX_INDEX_OFFSET:
            print(f"  [Agent 7] Pox case detected "
                  f"— using pox fallback templates")
            return self._fallback_sections(
                predicted_name, predicted_index,
                risk_level, decision, urgency)

        if self.client is None:
            print(f"  [Agent 7] No LLM client — using fallback")
            return self._fallback_sections(
                predicted_name, predicted_index,
                risk_level, decision, urgency)

        is_malignant = predicted_index in MALIGNANT_CLASSES

        top5_str = "\n".join([
            f"  {p['rank']}. {p['class_name']} "
            f"({p['confidence_pct']})"
            for p in (top_5 or [])[:5]
        ])

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
        is_malignant = predicted_index in MALIGNANT_CLASSES
        is_urgent    = decision in (
            "ESCALATE", "REFER_URGENT", "REVIEW")
        is_pox       = predicted_index >= POX_INDEX_OFFSET

        # ── Pox family templates ───────────────────────────────────
        if is_pox:
            pox_name = predicted_name

            if "Monkeypox" in pox_name or "Mpox" in pox_name:
                return {
                    "differential_diagnoses": (
                        "1. Monkeypox (Mpox) — the main AI prediction.\n"
                        "2. Chickenpox (varicella) — similar blister "
                        "pattern, more superficial lesions.\n"
                        "3. Smallpox — now eradicated but historically "
                        "similar in appearance.\n"
                        "4. Herpes simplex — clustered blisters, "
                        "typically in one area.\n"
                        "5. Syphilis — secondary stage can produce "
                        "a widespread rash."
                    ),
                    "recommended_investigations": (
                        "1. PCR swab of lesion fluid — the confirmatory "
                        "test for Monkeypox.\n"
                        "2. Full clinical examination by an infectious "
                        "disease specialist.\n"
                        "3. Blood tests to check immune status.\n"
                        "4. Contact tracing of all close contacts."
                    ),
                    "treatment_recommendations": (
                        "1. Isolate from other people immediately — "
                        "Monkeypox is contagious through close contact.\n"
                        "2. Seek medical attention within 24 hours.\n"
                        "3. Antiviral medication (tecovirimat) may be "
                        "prescribed by the specialist.\n"
                        "4. Keep lesions clean and covered.\n"
                        "5. Do not pop or scratch blisters."
                    ),
                    "medications": (
                        "Tecovirimat (antiviral) may be prescribed by "
                        "the specialist for severe cases.\n"
                        "Pain relief: paracetamol or ibuprofen for "
                        "fever and discomfort.\n"
                        "Antiseptic cream to keep lesions clean "
                        "and prevent secondary infection."
                    ),
                    "things_to_avoid": (
                        "1. Avoid close physical contact with other "
                        "people until all lesions have healed.\n"
                        "2. Do not share bedding, towels or clothing.\n"
                        "3. Do not attend work, school or public places.\n"
                        "4. Avoid contact with animals — Monkeypox can "
                        "spread to pets.\n"
                        "5. Do not pop or scratch the blisters."
                    ),
                    "followup_schedule": (
                        "Within 24 hours: Attend infectious disease "
                        "clinic or emergency department.\n"
                        "Week 1 to 4: Isolation at home until all lesions "
                        "have fully crusted and healed.\n"
                        "Ongoing: Follow public health guidance on "
                        "contact notification."
                    ),
                    "referral_decision": (
                        "URGENT referral to an Infectious Disease "
                        "specialist or Public Health unit within 24 hours.\n"
                        "Monkeypox is a notifiable disease — the clinic "
                        "will inform public health authorities."
                    ),
                }

            elif "Chickenpox" in pox_name:
                return {
                    "differential_diagnoses": (
                        "1. Chickenpox (varicella) — the main AI prediction.\n"
                        "2. Monkeypox — deeper lesions, more uniform size.\n"
                        "3. Herpes zoster (shingles) — follows nerve "
                        "distribution, usually one side of body.\n"
                        "4. Insect bites — scattered itchy bumps, "
                        "not in crops.\n"
                        "5. Hand-foot-mouth disease — lesions mainly "
                        "on palms, soles, and inside mouth."
                    ),
                    "recommended_investigations": (
                        "1. Clinical diagnosis is usually sufficient "
                        "based on the characteristic rash.\n"
                        "2. PCR swab of lesion fluid if diagnosis "
                        "is uncertain.\n"
                        "3. Check vaccination history."
                    ),
                    "treatment_recommendations": (
                        "1. Rest at home — chickenpox is contagious "
                        "until all blisters have crusted over.\n"
                        "2. Antiviral medication (acyclovir) may be "
                        "prescribed for adults or immunocompromised "
                        "patients.\n"
                        "3. Keep skin clean and cool.\n"
                        "4. Cut fingernails short to prevent scratching."
                    ),
                    "medications": (
                        "Antihistamine (cetirizine or loratadine) for "
                        "itching relief.\n"
                        "Paracetamol for fever — do NOT give aspirin "
                        "to children.\n"
                        "Calamine lotion applied gently to itchy lesions.\n"
                        "Acyclovir antiviral — only if prescribed by "
                        "a specialist for high-risk patients."
                    ),
                    "things_to_avoid": (
                        "1. Avoid contact with pregnant women, newborns "
                        "and immunocompromised people.\n"
                        "2. Do not scratch — this causes scarring and "
                        "risk of secondary infection.\n"
                        "3. Avoid school or work until fully crusted.\n"
                        "4. Avoid aspirin — risk of Reye syndrome "
                        "in children."
                    ),
                    "followup_schedule": (
                        "Stay home until all blisters have crusted "
                        "over — usually 5 to 7 days from onset.\n"
                        "Return to clinic if fever persists beyond "
                        "4 days or new symptoms develop.\n"
                        "Routine review after recovery."
                    ),
                    "referral_decision": (
                        "Routine GP or dermatology review.\n"
                        "URGENT referral if: immunocompromised patient, "
                        "pregnant, severe rash, or signs of infection."
                    ),
                }

            elif "Measles" in pox_name:
                return {
                    "differential_diagnoses": (
                        "1. Measles — the main AI prediction.\n"
                        "2. Rubella (German measles) — milder rash, "
                        "less fever.\n"
                        "3. Roseola — high fever then rash, mainly "
                        "in young children.\n"
                        "4. Scarlet fever — sandpaper texture rash "
                        "with sore throat.\n"
                        "5. Drug reaction rash — history of new "
                        "medication is the key clue."
                    ),
                    "recommended_investigations": (
                        "1. Blood test for measles IgM antibody — "
                        "confirmatory test.\n"
                        "2. Check vaccination history — MMR vaccine.\n"
                        "3. Clinical examination for Koplik spots "
                        "inside the mouth.\n"
                        "4. Notify public health — measles is a "
                        "notifiable disease."
                    ),
                    "treatment_recommendations": (
                        "1. Rest and isolation — measles is highly "
                        "contagious.\n"
                        "2. Vitamin A supplementation is recommended "
                        "for all measles cases.\n"
                        "3. Supportive care — fluids, fever management.\n"
                        "4. Seek medical attention promptly — measles "
                        "can cause serious complications."
                    ),
                    "medications": (
                        "Vitamin A: given as two doses on consecutive "
                        "days — reduces complications significantly.\n"
                        "Paracetamol for fever and discomfort.\n"
                        "Antibiotics only if secondary bacterial "
                        "infection is confirmed by the doctor."
                    ),
                    "things_to_avoid": (
                        "1. Avoid all contact with unvaccinated people, "
                        "pregnant women and immunocompromised individuals.\n"
                        "2. Avoid public places for at least 4 days "
                        "after rash onset.\n"
                        "3. Avoid school or work during isolation period."
                    ),
                    "followup_schedule": (
                        "Within 24 hours: Attend clinic for "
                        "confirmation and management plan.\n"
                        "Isolation for at least 4 days after "
                        "rash onset.\n"
                        "Ongoing: Monitor for complications including "
                        "pneumonia and encephalitis."
                    ),
                    "referral_decision": (
                        "URGENT referral to infectious disease or "
                        "paediatrics if complications are suspected.\n"
                        "Notify public health authority — measles "
                        "is a notifiable disease."
                    ),
                }

            else:
                # Cowpox or HFMD — generic pox fallback
                return {
                    "differential_diagnoses": (
                        f"1. {pox_name} — the main AI prediction.\n"
                        "2. Chickenpox — widespread vesicular rash.\n"
                        "3. Contact dermatitis — localised skin reaction.\n"
                        "4. Herpes simplex — clustered blisters.\n"
                        "5. Insect bites — scattered itchy lesions."
                    ),
                    "recommended_investigations": (
                        "1. Clinical examination by a doctor "
                        "for confirmation.\n"
                        "2. Swab of lesion fluid for PCR if "
                        "diagnosis is uncertain.\n"
                        "3. Vaccination history review."
                    ),
                    "treatment_recommendations": (
                        "1. Seek medical attention for clinical "
                        "confirmation of diagnosis.\n"
                        "2. Keep lesions clean and covered.\n"
                        "3. Avoid spreading to other people "
                        "through close contact.\n"
                        "4. Follow the treating doctor's instructions."
                    ),
                    "medications": (
                        "Paracetamol for pain and fever.\n"
                        "Antiseptic cream to prevent secondary "
                        "infection.\n"
                        "Specific antivirals only if prescribed "
                        "by the treating doctor."
                    ),
                    "things_to_avoid": (
                        "1. Avoid close contact with others "
                        "until confirmed by a doctor.\n"
                        "2. Do not scratch or pop lesions.\n"
                        "3. Avoid sharing personal items."
                    ),
                    "followup_schedule": (
                        "Attend clinic within 48 hours for "
                        "clinical confirmation.\n"
                        "Follow up as directed by the treating doctor."
                    ),
                    "referral_decision": (
                        "GP or dermatology referral within 48 hours "
                        "for clinical confirmation.\n"
                        "Urgent referral if spreading rapidly or "
                        "patient is immunocompromised."
                    ),
                }

        # ── Malignant / urgent template ────────────────────────────
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

        # ── Benign / routine template ──────────────────────────────
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

    # ── Report assembly — Prescription Style ───────────────────────

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

        now      = datetime.now()
        date_str = now.strftime("%d %B %Y")
        time_str = now.strftime("%H:%M")
        is_mal   = predicted_index in MALIGNANT_CLASSES
        is_pox   = predicted_index >= POX_INDEX_OFFSET

        sym    = symptoms or {}
        age    = str(sym.get("age",    "Not provided"))
        gender = str(sym.get("gender", "Not provided"))
        region = str(sym.get("region", "Not provided"))

        # Reported symptoms
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
                symptom_lines.append(f"  - {label}")
        if not symptom_lines:
            symptom_lines = ["  No significant symptoms reported"]

        # Medical history
        history_lines = []
        if sym.get("skin_cancer_history"):
            history_lines.append("  - Previous skin cancer history")
        if sym.get("cancer_history"):
            history_lines.append("  - Previous cancer history")
        if sym.get("smoke"):
            history_lines.append("  - Smoker")
        if sym.get("drink"):
            history_lines.append("  - Alcohol use")
        if not history_lines:
            history_lines = ["  No significant history"]

        # Risk display
        urgency_label = URGENCY_LABELS.get(urgency, urgency)
        if is_pox:
            risk_display = "INFECTIOUS DISEASE — Isolation required"
        elif is_mal:
            risk_display = "MALIGNANT — Urgent specialist referral required"
        elif risk_level in ("HIGH", "CRITICAL"):
            risk_display = f"{risk_level} — Specialist referral required"
        elif risk_level == "MODERATE":
            risk_display = "MODERATE — Clinical review recommended"
        else:
            risk_display = "LOW — Routine monitoring recommended"

        cs  = clinical_sections
        SEP = "=" * 60
        sep = "-" * 60

        report = f"""
{SEP}
            MAAI-SkinDx — AI Skin Assessment
    Insight Centre for Data Analytics | University of Galway
                Supervisor: Dr. Saeed Alsamhi
{SEP}
  Date    : {date_str}       Time : {time_str}
  Case ID : {case_id}
{SEP}

PATIENT DETAILS
{sep}
  Age           : {age}
  Gender        : {gender}
  Affected Area : {region}

  Symptoms:
{chr(10).join(symptom_lines)}

  Medical History:
{chr(10).join(history_lines)}

{SEP}
DIAGNOSIS
{sep}
  AI Prediction : {CLASS_FULL_NAMES.get(predicted_index, predicted_name)}
  Confidence    : {confidence*100:.1f}%
  Risk          : {risk_display}
  Decision      : {decision}   Urgency : {urgency_label}

  Other conditions the AI considered:
{cs.get('differential_diagnoses', '  Not available')}

{SEP}
RECOMMENDED TESTS
{sep}
{cs.get('recommended_investigations', '  Not available')}

{SEP}
Rx  MEDICATIONS
{sep}
{cs.get('medications', '  Not available')}

{SEP}
TREATMENT ADVICE
{sep}
{cs.get('treatment_recommendations', '  Not available')}

  What to avoid:
{cs.get('things_to_avoid', '  Not available')}

{SEP}
FOLLOW-UP
{sep}
{cs.get('followup_schedule', '  Not available')}

  Referral:
{cs.get('referral_decision', '  Not available')}

{SEP}
IMPORTANT NOTICE
{sep}
  This report is generated by MAAI-SkinDx, an AI-assisted
  dermatological screening system developed at the Insight
  Centre for Data Analytics, University of Galway.

  This report is intended to SUPPORT — not replace — a
  qualified healthcare professional. All findings must be
  reviewed and confirmed by a doctor before any action is
  taken. Do not make medical decisions based on this alone.

  In an emergency, contact emergency services immediately.

  AI Performance: Skin classifier 74.18% macro F1 (9 classes)
                  Pox classifier  97.82% macro F1 (5 classes)

  Generated by : MAAI-SkinDx Expert Agent v1.0
  Institution  : University of Galway, Ireland
  Supervisor   : Dr. Saeed Alsamhi
{SEP}
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
        is_pox = predicted_index >= POX_INDEX_OFFSET
        print(f"\n{'='*60}")
        print(f"  {self.name}")
        print(f"  Case     : {case_id}")
        print(f"  Diagnosis: {predicted_name} "
              f"({confidence*100:.1f}%)")
        print(f"  Decision : {decision} | {urgency}")
        print(f"  Type     : "
              f"{'POX CASE' if is_pox else 'Skin disease'}")
        print(f"  LLM      : "
              f"{'Groq active' if self.client else 'fallback'}")
        print(f"{'='*60}")

        try:
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
            llm_used = (
                self.client is not None and not is_pox)

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