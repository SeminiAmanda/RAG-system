import re
from pathlib import Path

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
QUESTIONS_PATH = Path("eval/questions.jsonl")

# Keep the index OUTSIDE OneDrive: Chroma uses SQLite, which can misbehave
# in synced folders. This puts it in ~/.rag-ds-ai/index
INDEX_DIR = Path.home() / ".rag-ds-ai" / "index"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# bge models work better when short queries get this prefix (documents get none)
QUERY_PREFIXES = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}

# Embed "heading path + text" instead of just text (an ablation you can toggle)
EMBED_WITH_HEADING = True


def collection_name(model_name: str = EMBED_MODEL) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]", "_", model_name)
    return f"chunks_{slug}" + ("_h" if EMBED_WITH_HEADING else "")


RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60