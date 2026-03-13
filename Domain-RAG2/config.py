import os

DOCS_FOLDER = os.getenv("DOCS_DIR", "docs")
ALLOWED_EXTENSIONS = [".pdf", ".txt"]

WORDS_PER_CHUNK = int(os.getenv("CHUNK_SIZE", "500"))
OVERLAP_WORDS   = int(os.getenv("CHUNK_OVERLAP", "50"))

EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FETCH_COUNT = int(os.getenv("TOP_K", "3"))

ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_VERSION = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
AI_ENABLED     = bool(ANTHROPIC_KEY)

LOG_PATH  = os.getenv("LOG_FILE", "logs/app.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

SCORE_CUTOFF = float(os.getenv("CONFIDENCE_THRESHOLD", "1.5"))
