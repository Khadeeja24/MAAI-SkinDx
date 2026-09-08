# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Build RAG Knowledge Base — Production Version
# ══════════════════════════════════════════════════════════════════
# Scrapes DermNet NZ articles for all 9 disease classes.
#
# All 5 fixes applied:
#   Fix 1 — Text cleaning: removes references, ads, navigation,
#            DOI strings, author lists before chunking
#   Fix 2 — Paragraph-based chunking: splits on paragraph and
#            sentence boundaries, never cuts mid-sentence
#   Fix 3 — Source diversity: max 8 chunks per URL stored
#   Fix 4 — Quality filter: rejects chunks with >40% reference text
#   Fix 5 — BioLORD-2023 biomedical embedding model: trained on
#            clinical ontologies, far better for dermatology terms
#
# Run ONCE before using Agent 5.
# ══════════════════════════════════════════════════════════════════

import os
import sys
import re
import time
import requests

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from bs4 import BeautifulSoup
except ImportError:
    os.system("pip install beautifulsoup4")
    from bs4 import BeautifulSoup

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction
)

# ── Configuration ──────────────────────────────────────────────────
CHROMA_DIR         = os.path.join(PROJECT_ROOT, "data", "chroma_db")
KB_DIR             = os.path.join(PROJECT_ROOT, "data", "knowledge_base")
COLLECTION_NAME    = "dermatology_knowledge"
EMBEDDING_MODEL    = "FremyCompany/BioLORD-2023"
CHUNK_SIZE         = 600
CHUNK_OVERLAP      = 80
MAX_CHUNKS_PER_URL = 8
MIN_CHUNK_LENGTH   = 150
QUALITY_THRESHOLD  = 0.40

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(KB_DIR,     exist_ok=True)

# ── DermNet URLs for all 9 disease classes ─────────────────────────
DERMNET_URLS = {
    "Melanoma": [
        "https://dermnetnz.org/topics/melanoma",
        "https://dermnetnz.org/topics/superficial-spreading-melanoma",
        "https://dermnetnz.org/topics/nodular-melanoma",
        "https://dermnetnz.org/topics/melanoma-overview",
    ],
    "Melanocytic Nevus": [
        "https://dermnetnz.org/topics/melanocytic-naevus",
        "https://dermnetnz.org/topics/atypical-naevus",
        "https://dermnetnz.org/topics/congenital-melanocytic-naevus",
    ],
    "Basal Cell Carcinoma": [
        "https://dermnetnz.org/topics/basal-cell-carcinoma",
        "https://dermnetnz.org/topics/morphoeic-basal-cell-carcinoma",
        "https://dermnetnz.org/topics/superficial-basal-cell-carcinoma",
    ],
    "Actinic Keratosis": [
        "https://dermnetnz.org/topics/actinic-keratosis",
        "https://dermnetnz.org/topics/squamous-cell-carcinoma-overview",
        "https://dermnetnz.org/topics/bowen-disease",
    ],
    "Benign Keratosis": [
        "https://dermnetnz.org/topics/seborrhoeic-keratosis",
        "https://dermnetnz.org/topics/stucco-keratosis",
        "https://dermnetnz.org/topics/dermatosis-papulosa-nigra",
    ],
    "Vascular Lesion": [
        "https://dermnetnz.org/topics/pyogenic-granuloma",
        "https://dermnetnz.org/topics/cherry-angioma",
        "https://dermnetnz.org/topics/spider-naevus",
        "https://dermnetnz.org/topics/venous-lake",
    ],
    "Dermatofibroma": [
        "https://dermnetnz.org/topics/dermatofibroma",
        "https://dermnetnz.org/topics/fibrous-skin-tumours",
    ],
    "Inflammatory": [
        "https://dermnetnz.org/topics/psoriasis",
        "https://dermnetnz.org/topics/lichen-planus",
        "https://dermnetnz.org/topics/pityriasis-rosea",
        "https://dermnetnz.org/topics/seborrhoeic-dermatitis",
    ],
    "Eczema / Dermatitis": [
        "https://dermnetnz.org/topics/atopic-dermatitis",
        "https://dermnetnz.org/topics/contact-dermatitis",
        "https://dermnetnz.org/topics/nummular-dermatitis",
    ],
}

print(f"\n{'='*65}")
print(f"  MAAI-SkinDx | Building RAG Knowledge Base (Production)")
print(f"{'='*65}")
print(f"  Embedding model : {EMBEDDING_MODEL}")
print(f"  Fix 1           : Text cleaning")
print(f"  Fix 2           : Paragraph-based chunking")
print(f"  Fix 3           : Source diversity cap ({MAX_CHUNKS_PER_URL}/URL)")
print(f"  Fix 4           : Quality filter (>{QUALITY_THRESHOLD*100:.0f}% refs = discard)")
print(f"  Fix 5           : BioLORD-2023 biomedical embeddings")
print(f"{'='*65}\n")

# ══════════════════════════════════════════════════════════════════
# FIX 1 — Text cleaning
# Remove non-clinical content before chunking
# ══════════════════════════════════════════════════════════════════

REFERENCE_PATTERNS = [
    r'\b\d{4};\d+\(\d+\):\d+',
    r'\bdoi:\s*10\.\d{4}',
    r'\bPubMed\b',
    r'\bMedline\b',
    r'\bOn DermNet\b',
    r'\bADVERTISEMENT\b',
    r'\bReferences\b',
    r'et al\.',
    r'\bJ [A-Z][a-z]+ [A-Z][a-z]+\b',
    r'\bDermatol\.\b',
    r'\bSkin Cancer Foundation\b',
    r'\bAmerican Academy of Dermatology\b',
    r'Copyright|©|\u00a9',
    r'Books about skin',
    r'Other websites',
    r'Introduction Demographics',
]
REFERENCE_RE = re.compile(
    '|'.join(REFERENCE_PATTERNS), re.IGNORECASE)


def clean_text(text: str) -> str:
    """
    Fix 1: Remove non-clinical content.
    Keeps: clinical descriptions, symptoms, management, diagnosis.
    Removes: references, DOIs, ads, navigation, author lists.
    """
    text      = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)

    clean = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        ref_matches = len(REFERENCE_RE.findall(s))
        words       = len(s.split())
        if words > 0 and ref_matches / max(words, 1) > 0.3:
            continue
        if re.match(r'^[A-Z][a-z]+\s[A-Z]{1,2},', s):
            continue
        if s.startswith('doi:') or 'doi.org' in s:
            continue
        clean.append(s)

    return ' '.join(clean)


# ══════════════════════════════════════════════════════════════════
# FIX 4 — Chunk quality filter
# Rejects chunks with too much reference-style content
# ══════════════════════════════════════════════════════════════════

CLINICAL_WORDS = [
    'lesion', 'skin', 'treatment', 'diagnosis', 'clinical',
    'patient', 'dermat', 'cancer', 'benign', 'malignant',
    'symptoms', 'causes', 'management', 'biopsy', 'melanoma',
    'nevus', 'naevus', 'keratosis', 'carcinoma', 'eczema',
    'inflammation', 'pigment', 'colour', 'border', 'asymmetr',
    'dermoscop', 'histol', 'excision', 'monitor', 'referral',
]


def is_quality_chunk(text: str) -> bool:
    """
    Fix 4: Return True only if chunk has sufficient clinical content.
    """
    if len(text.strip()) < MIN_CHUNK_LENGTH:
        return False
    words       = text.split()
    ref_matches = len(REFERENCE_RE.findall(text))
    if len(words) == 0:
        return False
    if ref_matches / len(words) > QUALITY_THRESHOLD:
        return False
    text_lower   = text.lower()
    has_clinical = any(w in text_lower for w in CLINICAL_WORDS)
    return has_clinical


# ══════════════════════════════════════════════════════════════════
# FIX 2 — Paragraph-based chunking
# Splits on paragraph and sentence boundaries
# ══════════════════════════════════════════════════════════════════

def chunk_by_paragraph(
    text    : str,
    source  : str,
    disease : str
) -> list:
    """
    Fix 2 — Sentence-boundary chunking for web-scraped text.
    DermNet articles have no paragraph markers after scraping.
    Split on sentence boundaries instead. Group sentences into
    chunks of approximately CHUNK_SIZE characters with overlap.
    Clinical sentences never cut in the middle.
    """
    # Split into sentences on period + space
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    chunks  = []
    chunk_n = 0
    buffer  = ""

    for sentence in sentences:
        # If adding this sentence keeps buffer under limit — accumulate
        if len(buffer) + len(sentence) + 1 <= CHUNK_SIZE:
            buffer = (buffer + " " + sentence).strip()

        else:
            # Save current buffer as chunk
            if len(buffer) >= MIN_CHUNK_LENGTH:
                chunks.append({
                    "text"    : buffer,
                    "disease" : disease,
                    "source"  : source,
                    "chunk_id": chunk_n,
                })
                chunk_n += 1

            # Start new buffer with overlap
            # Keep last 2 sentences for context continuity
            if buffer:
                prev_sentences = re.split(
                    r'(?<=[.!?])\s+', buffer)
                overlap = " ".join(
                    prev_sentences[-2:]
                ) if len(prev_sentences) >= 2 else ""
                buffer = (overlap + " " + sentence).strip()
            else:
                buffer = sentence

    # Save remaining buffer
    if len(buffer) >= MIN_CHUNK_LENGTH:
        chunks.append({
            "text"    : buffer,
            "disease" : disease,
            "source"  : source,
            "chunk_id": chunk_n,
        })

    return chunks


# ── Scraping ───────────────────────────────────────────────────────

def scrape_dermnet(url: str) -> str:
    """Scrape clinical text from DermNet NZ article."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f"    WARNING: {url} returned {resp.status_code}")
            return ""

        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup([
            "nav", "footer", "script", "style",
            "header", "aside", "form", "button",
            "noscript", "iframe", "img"
        ]):
            tag.decompose()

        main = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", class_=lambda c:
                c and any(x in c.lower()
                          for x in ["content", "article",
                                     "main", "body"])
            ) or soup.body
        )
        if main is None:
            return ""

        text = main.get_text(separator=" ", strip=True)
        return re.sub(r'\s+', ' ', text).strip()

    except Exception as e:
        print(f"    ERROR scraping {url}: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════
# Build knowledge base
# ══════════════════════════════════════════════════════════════════

print("── Step 1: Initialising ChromaDB ──")
db_client = chromadb.PersistentClient(path=CHROMA_DIR)

try:
    db_client.delete_collection(COLLECTION_NAME)
    print(f"  Deleted existing collection")
except Exception:
    pass

ef = SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL)
collection = db_client.create_collection(
    name               = COLLECTION_NAME,
    embedding_function = ef,
    metadata           = {"hnsw:space": "cosine"}
)
print(f"  Collection created : {COLLECTION_NAME}")
print(f"  Embedding model    : {EMBEDDING_MODEL}\n")

print("── Step 2: Scraping, Cleaning, Chunking, Embedding ──")

doc_id          = 0
total_stored    = 0
summary_rows    = []

for disease, urls in DERMNET_URLS.items():
    print(f"\n  Disease: {disease}")
    disease_stored = 0

    for url in urls:
        print(f"    Scraping  : {url}")
        raw_text = scrape_dermnet(url)

        if not raw_text:
            print(f"    SKIP — no content retrieved")
            continue

        # Fix 1 — Clean
        clean = clean_text(raw_text)
        print(f"    Raw→Clean : {len(raw_text):,} → "
              f"{len(clean):,} chars "
              f"({100*len(clean)//max(len(raw_text),1)}% kept)")

        if len(clean) < MIN_CHUNK_LENGTH:
            print(f"    SKIP — insufficient content after cleaning")
            continue

        # Fix 2 — Paragraph chunking
        chunks = chunk_by_paragraph(clean, url, disease)
        print(f"    Chunks    : {len(chunks)} (paragraph-based)")

        # Fix 4 — Quality filter
        quality = [c for c in chunks if is_quality_chunk(c["text"])]
        dropped = len(chunks) - len(quality)
        if dropped > 0:
            print(f"    Quality   : {len(quality)} kept, "
                  f"{dropped} discarded")

        # Fix 3 — Source diversity cap
        capped           = quality[:MAX_CHUNKS_PER_URL]
        diversity_drop   = len(quality) - len(capped)
        if diversity_drop > 0:
            print(f"    Diversity : capped at {MAX_CHUNKS_PER_URL} "
                  f"({diversity_drop} dropped)")

        # Store in ChromaDB
        for chunk in capped:
            collection.add(
                documents = [chunk["text"]],
                metadatas = [{
                    "disease" : chunk["disease"],
                    "source"  : chunk["source"],
                    "chunk_id": str(chunk["chunk_id"]),
                }],
                ids = [f"doc_{doc_id}"]
            )
            doc_id         += 1
            disease_stored += 1
            total_stored   += 1

        print(f"    Stored    : {len(capped)} chunks")
        time.sleep(1.2)

    print(f"  Total stored for {disease}: {disease_stored}")
    summary_rows.append((disease, disease_stored))

total_final = collection.count()

print(f"\n{'='*65}")
print(f"  KNOWLEDGE BASE COMPLETE")
print(f"{'='*65}")
print(f"  Embedding model : {EMBEDDING_MODEL}")
print(f"  Total stored    : {total_final} chunks")
print(f"  ChromaDB path   : {CHROMA_DIR}")
print(f"\n  Per-class summary:")
for disease, count in summary_rows:
    bar = '█' * min(count, 30)
    print(f"    {disease:<25} : {count:>3}  {bar}")
print(f"\n  Next: python notebooks/test_orchestrator.py")
print(f"{'='*65}\n")