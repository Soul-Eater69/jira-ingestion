import os
from dotenv import load_dotenv

load_dotenv()

# --- Jira ---
JIRA_BASE_URL: str = os.environ.get("JIRA_BASE_URL", "https://jira.fyiblue.com")
JIRA_TOKEN: str = os.environ.get("JIRA_API_TOKEN", "")
JIRA_VERIFY_SSL: bool = os.environ.get("JIRA_VERIFY_SSL", "true").lower() == "true"

# --- LLM ---
LLM_MODEL: str = os.environ.get("LLM_MODEL", "gpt-4o-mini")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# --- Embeddings ---
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")

# --- Indexes ---
COARSE_INDEX_NAME: str = os.environ.get("COARSE_INDEX_NAME", "tickets_coarse")
FINE_INDEX_NAME: str = os.environ.get("FINE_INDEX_NAME", "tickets_fine")
SUPERVISION_STORE: str = os.environ.get("SUPERVISION_STORE", "tickets_supervision")
PINECONE_API_KEY: str = os.environ.get("PINECONE_API_KEY", "")
PINECONE_ENVIRONMENT: str = os.environ.get("PINECONE_ENVIRONMENT", "us-east-1")

# --- Entity dictionaries ---
ENTITY_DICT_PATH: str = os.environ.get("ENTITY_DICT_PATH", "data/entity_dicts/")
ENTITY_DICT_REFRESH: str = os.environ.get("ENTITY_DICT_REFRESH", "daily")

# --- Extraction ---
OCR_ENABLED: bool = os.environ.get("OCR_ENABLED", "true").lower() == "true"
MAX_SLIDES: int = int(os.environ.get("MAX_SLIDES", "60"))
MAX_SUPPLEMENTARY: int = int(os.environ.get("MAX_SUPPLEMENTARY", "2"))
TABLE_SUMMARY_CACHE_TTL: int = int(os.environ.get("TABLE_SUMMARY_CACHE_TTL_SECONDS", str(7 * 24 * 3600)))

# --- Triage thresholds ---
MIN_FILE_SIZE_BYTES: int = 15_000
MIN_PDF_SIZE_BYTES: int = 50_000
MAX_FILE_SIZE_BYTES: int = 100_000_000
LAYER1_SKIP_PEEK_SCORE: int = 60
LAYER1_SKIP_PEEK_GAP: int = 20

# --- Description thresholds ---
DESC_JUNK_MAX_WORDS: int = 10
DESC_THIN_MAX_WORDS: int = 50
DESC_RICH_MIN_WORDS: int = 150

# --- Section chunking ---
SECTION_MIN_SLIDES: int = 8

# --- Entity extraction ---
ENTITY_CONFIDENCE_TEXT_MATCH: float = 0.8
ENTITY_CONFIDENCE_COMPONENT_MATCH: float = 1.0
