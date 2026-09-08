# ══════════════════════════════════════════════════════════════════
# MAAI-SkinDx | Agent 5 — RAG Configuration
# ══════════════════════════════════════════════════════════════════

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
               os.path.abspath(__file__))))

# Knowledge base storage
KB_DIR       = os.path.join(PROJECT_ROOT, "data", "knowledge_base")
CHROMA_DIR   = os.path.join(PROJECT_ROOT, "data", "chroma_db")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs", "agent5_rag")

os.makedirs(KB_DIR,     exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Embedding model — runs locally, no API key needed
EMBEDDING_MODEL = "FremyCompany/BioLORD-2023"
COLLECTION_NAME  = "dermatology_knowledge"

# Retrieval settings
TOP_K_CHUNKS     = 5      # number of chunks retrieved per query
CHUNK_SIZE       = 800    # characters per chunk
CHUNK_OVERLAP    = 100    # overlap between chunks

# LLM settings
LLM_MODEL        = "openai/gpt-oss-120b"
LLM_MAX_TOKENS   = 600
LLM_TEMPERATURE  = 0.2

# DermNet URLs for all 9 disease classes
DERMNET_URLS = {
    "Melanoma": [
        "https://dermnetnz.org/topics/melanoma",
        "https://dermnetnz.org/topics/melanoma-diagnosis",
        "https://dermnetnz.org/topics/melanoma-treatment",
    ],
    "Melanocytic Nevus": [
        "https://dermnetnz.org/topics/melanocytic-naevus",
        "https://dermnetnz.org/topics/dysplastic-naevus",
    ],
    "Basal Cell Carcinoma": [
        "https://dermnetnz.org/topics/basal-cell-carcinoma",
        "https://dermnetnz.org/topics/basal-cell-carcinoma-treatment",
    ],
    "Actinic Keratosis": [
        "https://dermnetnz.org/topics/actinic-keratosis",
        "https://dermnetnz.org/topics/squamous-cell-carcinoma",
    ],
    "Benign Keratosis": [
        "https://dermnetnz.org/topics/seborrhoeic-keratosis",
    ],
    "Vascular Lesion": [
    "https://dermnetnz.org/topics/port-wine-stain",
    "https://dermnetnz.org/topics/pyogenic-granuloma",
    "https://dermnetnz.org/topics/cherry-angioma",
    ],
    "Dermatofibroma": [
        "https://dermnetnz.org/topics/dermatofibroma",
    ],
    "Inflammatory": [
        "https://dermnetnz.org/topics/psoriasis",
        "https://dermnetnz.org/topics/lichen-planus",
    ],
    "Eczema / Dermatitis": [
        "https://dermnetnz.org/topics/atopic-dermatitis",
        "https://dermnetnz.org/topics/contact-dermatitis",
    ],
}

# Class index to name mapping
CLASS_NAMES = {
    0: "Melanoma",
    1: "Melanocytic Nevus",
    2: "Basal Cell Carcinoma",
    3: "Actinic Keratosis",
    4: "Benign Keratosis",
    5: "Vascular Lesion",
    6: "Dermatofibroma",
    7: "Inflammatory",
    8: "Eczema / Dermatitis",
}