# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 5 — RAG Knowledge Retrieval Agent
# ══════════════════════════════════════════════════════════════════
# Receives case details from Orchestrator after Agent 4
# and Clinical Agent both complete.
#
# Two-part retrieval:
#   Part 1 — Vector retrieval from DermNet knowledge base
#             BioLORD-2023 biomedical embeddings
#             Source diversity enforced — max 2 chunks per URL
#             Clean name matching — strips Agent 3 description suffix
#   Part 2 — LLM synthesis via Groq openai/gpt-oss-120b
#             Synthesises retrieved knowledge + case context
#             into structured clinical knowledge summary
#             Falls back to raw chunks if LLM unavailable
#
# Reports back to Orchestrator only.
# Never communicates with any other agent directly.
# ══════════════════════════════════════════════════════════════════

import os
import sys
import traceback
import warnings
warnings.filterwarnings("ignore")

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .config import (
    CHROMA_DIR, EMBEDDING_MODEL, COLLECTION_NAME,
    TOP_K_CHUNKS, OUTPUT_DIR, LLM_MODEL,
    LLM_MAX_TOKENS, LLM_TEMPERATURE, CLASS_NAMES
)

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

try:
    import chromadb
    from chromadb.utils.embedding_functions import (
        SentenceTransformerEmbeddingFunction
    )
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Max chunks from any single source URL — Fix 3 source diversity
MAX_CHUNKS_PER_SOURCE = 2


class RAGAgent:

    def __init__(self):
        self.name       = "Agent 5 — RAG Knowledge Retrieval Agent"
        self.collection = None
        self.client     = None
        print(f"\n[{self.name}] Initialising ...")
        self._load_knowledge_base()
        self._init_llm()
        print(f"[{self.name}] Ready")

    # ─── Initialisation ────────────────────────────────────────────

    def _load_knowledge_base(self):
        """Load ChromaDB knowledge base from disk."""
        if not CHROMA_AVAILABLE:
            print(f"  [Agent 5] ChromaDB not installed")
            return
        if not os.path.exists(CHROMA_DIR):
            print(f"  [Agent 5] WARNING: Knowledge base not found")
            print(f"  Run: python scripts/build_knowledge_base.py")
            return
        try:
            ef = SentenceTransformerEmbeddingFunction(
                model_name=EMBEDDING_MODEL)
            db_client       = chromadb.PersistentClient(
                path=CHROMA_DIR)
            self.collection = db_client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=ef)
            count = self.collection.count()
            print(f"  [Agent 5] Knowledge base loaded")
            print(f"  [Agent 5] Chunks     : {count:,}")
            print(f"  [Agent 5] Embeddings : {EMBEDDING_MODEL}")
        except Exception as e:
            print(f"  [Agent 5] KB load failed: {e}")
            self.collection = None

    def _init_llm(self):
        """Initialise Groq LLM client."""
        if not GROQ_AVAILABLE:
            print(f"  [Agent 5] Groq not available")
            return
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            print(f"  [Agent 5] GROQ_API_KEY not set")
            return
        try:
            self.client = Groq(api_key=api_key)
            print(f"  [Agent 5] LLM: Groq {LLM_MODEL} ready")
        except Exception:
            self.client = None

    # ─── Retrieval ─────────────────────────────────────────────────

    def _clean_disease_name(self, predicted_name: str) -> str:
        """
        Agent 3 appends description text to the class name.
        Example: "Melanocytic Nevus — common benign mole"
        The ChromaDB metadata stores only "Melanocytic Nevus".
        Split on em dash and take the first part only.
        """
        return predicted_name.split("\u2014")[0].strip()

    def _apply_source_diversity(self, pairs: list) -> list:
        """
        Fix 3 — Source diversity enforcement.
        Limit to MAX_CHUNKS_PER_SOURCE chunks per source URL.
        Prevents one article dominating all retrieval results.
        Without this, all 5 chunks can come from the same URL.
        """
        seen_sources  = {}
        diverse_pairs = []
        for doc, meta in pairs:
            src   = meta.get("source", "")
            count = seen_sources.get(src, 0)
            if count < MAX_CHUNKS_PER_SOURCE:
                diverse_pairs.append((doc, meta))
                seen_sources[src] = count + 1
        return diverse_pairs

    def _retrieve(
        self,
        predicted_name : str,
        confidence     : float,
        risk_level     : str
    ) -> list:
        """
        Retrieve top-K clinically relevant chunks from ChromaDB.

        Query construction: disease name + clinical terms +
        risk level. This gives BioLORD context to find the
        most relevant clinical management content.

        Two-stage retrieval:
          Stage 1: filter by disease name (exact match on metadata)
          Stage 2: fallback without filter if Stage 1 returns nothing
        """
        if self.collection is None:
            return []

        clean_name = self._clean_disease_name(predicted_name)

        query = (
            f"{clean_name} clinical features dermoscopy "
            f"differential diagnosis management "
            f"treatment {risk_level} risk"
        )

        # Stage 1 — filtered retrieval with disease match
        try:
            results = self.collection.query(
                query_texts = [query],
                n_results   = TOP_K_CHUNKS,
                where       = {"disease": clean_name},
            )
            docs  = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            pairs = list(zip(docs, metas))

            if pairs:
                diverse = self._apply_source_diversity(pairs)
                if diverse:
                    return diverse

        except Exception as e:
            print(f"  [Agent 5] Filtered retrieval error: {e}")

        # Stage 2 — fallback without disease filter
        try:
            results = self.collection.query(
                query_texts = [query],
                n_results   = TOP_K_CHUNKS,
            )
            docs  = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            pairs = list(zip(docs, metas))
            return self._apply_source_diversity(pairs)

        except Exception as e2:
            print(f"  [Agent 5] Fallback retrieval error: {e2}")
            return []

    # ─── LLM Synthesis ─────────────────────────────────────────────

    def _synthesise(
        self,
        retrieved_chunks : list,
        predicted_name   : str,
        confidence       : float,
        top_5            : list,
        risk_level       : str,
        risk_score       : int,
        ml_probability   : float,
        focus_type       : str,
        xai_explanation  : str,
        case_id          : str,
    ) -> str:
        """
        Synthesise retrieved knowledge + case context via LLM.
        Falls back to raw chunks if LLM unavailable.

        The LLM receives:
          - AI diagnostic findings (Agent 3 output)
          - Clinical risk (Clinical Agent output)
          - Grad-CAM focus type (Agent 4 output)
          - XAI explanation excerpt (Agent 4 output)
          - Retrieved DermNet clinical knowledge chunks

        It produces a 4-5 sentence clinical summary that:
          1. Contextualises the AI prediction with clinical knowledge
          2. Notes key differential diagnoses
          3. Recommends management based on risk level
          4. Flags discordance between image and symptom findings
        """
        if self.client is None or not retrieved_chunks:
            if retrieved_chunks:
                return "\n\n".join(
                    [doc for doc, _ in retrieved_chunks[:3]])
            return "Knowledge base not available."

        # Build context — label each source
        context = "\n\n".join([
            f"[Source: {meta.get('source', 'DermNet')}]\n{doc}"
            for doc, meta in retrieved_chunks
        ])

        top5_str = "\n".join([
            f"  {p['rank']}. {p['class_name']} — "
            f"{p['confidence_pct']}"
            for p in (top_5 or [])[:5]
        ])

        ml_str = (
            f"{ml_probability*100:.1f}%"
            if ml_probability and ml_probability >= 0
            else "Not available"
        )

        system_prompt = (
            "You are a clinical dermatology knowledge assistant "
            "supporting qualified dermatologists. "
            "You synthesise retrieved clinical knowledge with "
            "AI diagnostic findings into a structured clinical "
            "summary. Be specific, evidence-based, and concise. "
            "Write for a qualified clinician, not a patient. "
            "Flag any discordance between image findings and "
            "clinical risk explicitly. Maximum 5 sentences."
        )

        user_prompt = (
            f"An AI system has analysed a skin lesion.\n\n"
            f"AI DIAGNOSTIC FINDINGS (Agent 3):\n"
            f"  Predicted condition : {predicted_name}\n"
            f"  Confidence          : {confidence*100:.1f}%\n\n"
            f"CLINICAL RISK (Clinical Agent):\n"
            f"  Risk level          : {risk_level}\n"
            f"  Rule-based score    : {risk_score}/100\n"
            f"  ML malignancy prob  : {ml_str}\n\n"
            f"XAI FINDINGS (Agent 4):\n"
            f"  Grad-CAM focus      : {focus_type}\n"
            f"  Explanation excerpt : "
            f"{xai_explanation[:300] if xai_explanation else 'Not available'}\n\n"
            f"TOP 5 PREDICTIONS:\n{top5_str}\n\n"
            f"RETRIEVED CLINICAL KNOWLEDGE (DermNet NZ):\n"
            f"{context}\n\n"
            f"Write a 4-5 sentence clinical knowledge summary:\n"
            f"1. Confirm or contextualise the AI prediction "
            f"using retrieved clinical knowledge\n"
            f"2. Note key differential diagnoses to consider\n"
            f"3. Recommend management based on risk level\n"
            f"4. Flag discordance if image prediction and "
            f"clinical risk tell different stories\n"
            f"5. State urgency clearly if risk is HIGH or CRITICAL"
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
            print(f"  [Agent 5] LLM error: {e}")
            return "\n\n".join(
                [doc for doc, _ in retrieved_chunks[:2]])

    # ─── Report Saving ─────────────────────────────────────────────

    def _save_report(
        self,
        case_id        : str,
        summary        : str,
        retrieved      : list,
        predicted_name : str,
        risk_level     : str,
    ) -> str:
        """Save RAG knowledge summary to disk."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        report_path = os.path.join(
            OUTPUT_DIR, f"{case_id}_rag_summary.txt")

        sources = list(set([
            meta.get("source", "DermNet")
            for _, meta in retrieved
        ]))

        content = (
            f"MAAI-SkinDx | Agent 5 \u2014 RAG Knowledge Summary\n"
            f"Case ID            : {case_id}\n"
            f"Embedding model    : {EMBEDDING_MODEL}\n"
            f"{'='*60}\n\n"
            f"PREDICTED CONDITION : {predicted_name}\n"
            f"CLINICAL RISK LEVEL : {risk_level}\n\n"
            f"CLINICAL KNOWLEDGE SUMMARY\n"
            f"  {summary}\n\n"
            f"KNOWLEDGE SOURCES ({len(retrieved)} chunks retrieved)\n"
        )
        for s in sources:
            content += f"  - {s}\n"
        content += f"\n{'='*60}\n"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)

        return report_path

    # ─── Main Entry Point ──────────────────────────────────────────

    def run(
        self,
        case_id          : str,
        predicted_index  : int,
        predicted_name   : str,
        confidence       : float,
        top_5            : list,
        risk_level       : str   = None,
        risk_score       : int   = None,
        ml_probability   : float = None,
        focus_type       : str   = None,
        xai_explanation  : str   = None,
    ) -> dict:
        """
        Retrieve clinical knowledge for predicted condition.
        Called by Orchestrator after Agent 4 and
        Clinical Agent both complete.
        """
        print(f"\n{'='*60}")
        print(f"  {self.name}")
        print(f"  Case      : {case_id}")
        print(f"  Condition : {predicted_name} "
              f"({confidence*100:.1f}%)")
        print(f"  Risk      : {risk_level or 'Not provided'}")
        print(f"{'='*60}")

        if self.collection is None:
            return {
                "status"          : "FAIL",
                "reason"          : "Knowledge base not loaded.",
                "summary"         : None,
                "report_path"     : None,
                "chunks_retrieved": 0,
                "sources"         : [],
            }

        # Step 1 — Retrieve
        print(f"\n  -- Retrieving from Knowledge Base --")
        retrieved = self._retrieve(
            predicted_name,
            confidence,
            risk_level or "UNKNOWN"
        )
        print(f"  Chunks retrieved : {len(retrieved)}")

        # Show source diversity
        source_counts = {}
        for _, meta in retrieved:
            src = meta.get("source", "Unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
        for src, count in source_counts.items():
            print(f"  Source ({count}): {src}")

        if not retrieved:
            return {
                "status"          : "FAIL",
                "reason"          : "No relevant chunks found.",
                "summary"         : None,
                "report_path"     : None,
                "chunks_retrieved": 0,
                "sources"         : [],
            }

        # Step 2 — LLM synthesis
        print(f"\n  -- Synthesising Clinical Summary --")
        summary = self._synthesise(
            retrieved_chunks = retrieved,
            predicted_name   = predicted_name,
            confidence       = confidence,
            top_5            = top_5,
            risk_level       = risk_level or "UNKNOWN",
            risk_score       = risk_score or 0,
            ml_probability   = ml_probability or -1.0,
            focus_type       = focus_type or "Unknown",
            xai_explanation  = xai_explanation or "",
            case_id          = case_id,
        )
        print(f"  Summary generated")

        # Step 3 — Save report
        report_path = self._save_report(
            case_id, summary, retrieved,
            predicted_name, risk_level or "UNKNOWN")
        print(f"  Report saved : {report_path}")
        print(f"{'='*60}\n")

        sources = list(set([
            meta.get("source", "DermNet")
            for _, meta in retrieved
        ]))

        return {
            "status"          : "PASS",
            "reason"          : "RAG knowledge retrieval complete.",
            "summary"         : summary,
            "report_path"     : report_path,
            "chunks_retrieved": len(retrieved),
            "sources"         : sources,
        }