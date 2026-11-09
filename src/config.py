import os
import re
from pathlib import Path

# Environment variables let experiments override these without editing the file.
CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", "data/processed/chunks.jsonl"))
QUESTIONS_PATH = Path("eval/questions.jsonl")

# Keep the index OUTSIDE OneDrive: Chroma uses SQLite, which can misbehave
# in synced folders. This puts it in ~/.rag-ds-ai/index
INDEX_DIR = Path.home() / ".rag-ds-ai" / "index"

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# bge models work better when short queries get this prefix (documents get none)
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
    "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

# Embed "heading path + text" instead of just text
EMBED_WITH_HEADING = os.getenv("EMBED_WITH_HEADING", "1") == "1"

# Names a chunking variant (e.g. c100); empty for the baseline chunking
CHUNK_TAG = os.getenv("CHUNK_TAG", "")

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60


def collection_name(model_name: str = EMBED_MODEL) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", model_name)
    name = f"chunks_{slug}" + ("_h" if EMBED_WITH_HEADING else "")
    return name + (f"_{CHUNK_TAG}" if CHUNK_TAG else "")