"""Application and grading configuration."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

# The weights are normalized again by the grading engine, so they may be tuned
# without risking marks outside the valid 0-maximum range.
FEATURE_WEIGHTS = {
    "semantic_similarity": 0.65,
    "tfidf_similarity": 0.10,
    "concept_coverage": 0.20,
    "completeness": 0.05,
}

SEMANTIC_MODEL_NAME = "all-MiniLM-L6-v2"
CONCEPT_SIMILARITY_THRESHOLD = 0.48
# Sparse lexical fallbacks understate meaning overlap compared with sentence embeddings.
# This monotonic calibration is used only when the local embedding model is unavailable.
SEMANTIC_FALLBACK_POWER = 0.30