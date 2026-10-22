"""
Central paths and defaults for ContextPilot.
Edit here instead of scattering magic strings.
"""
from pathlib import Path

# --- paths (repo root = parent of app/) ---
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CHARTS_DIR = DATA_DIR / "charts"
METADATA_DIR = DATA_DIR / "metadata"
CHART_METADATA_CSV = METADATA_DIR / "chart_metadata.csv"
QUERY_TRIPLETS_CSV = METADATA_DIR / "query_context_triplets.csv"

CACHE_DIR = ROOT_DIR / "cache"
EMBEDDINGS_DIR = CACHE_DIR / "embeddings"
FAISS_DIR = CACHE_DIR / "faiss"
RESULTS_DIR = CACHE_DIR / "results"

# --- chart metadata columns (must match CSV) ---
COL_FILENAME = "filename"
COL_CHART_TYPE = "chart_type"
COL_QUERY_INTENT = "query_intent"
COL_RELEVANT_TABLES = "relevant_tables"
COL_DESCRIPTION = "description"

# --- models ---
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_EMBED_DIM = 512

# baselines (Analysis page)
TEXT_BASELINE_ID = "distilbert-base-uncased"
IMAGE_BASELINE_ID = "mobilenet_v2"

# --- retrieval / fusion ---
DEFAULT_TOP_K = 5
FUSION_WEIGHT_TEXT = 0.55
FUSION_WEIGHT_VISION = 0.45

# --- token budget (Playground + LLM pack) ---
DEFAULT_TOKEN_BUDGET = 500
MIN_TOKEN_BUDGET = 200
MAX_TOKEN_BUDGET = 2000
TIKTOKEN_ENCODING = "cl100k_base"

# --- LLM ---
GROQ_MODEL = "qwen/qwen3.8-27b"
GEMINI_MODEL = "models/gemini-2.5-flash"
LLM_TEMPERATURE = 0.2
LLM_MAX_OUTPUT_TOKENS = 512

# --- UI ---
APP_TITLE = "ContextPilot"
APP_TAGLINE = "Query–chart context fusion for grounded answers"
