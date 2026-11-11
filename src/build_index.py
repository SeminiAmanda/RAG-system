"""Build the Chroma index from the committed corpus and embeddings (no embedding model needed)."""
import json
import os
from pathlib import Path

import chromadb
import numpy as np

from config import CHUNKS_PATH, INDEX_DIR, collection_name

EMBEDDINGS_PATH = Path(os.getenv("EMBEDDINGS_PATH", str(Path(CHUNKS_PATH).with_name("embeddings.npy"))))


def ensure_index() -> int:
    """Make sure the Chroma collection exists; build it from the saved embeddings if not."""
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    name = collection_name()
    try:
        col = client.get_collection(name)
        if col.count() > 0:
            return col.count()
    except Exception:
        pass  # the collection does not exist yet

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    emb = np.load(EMBEDDINGS_PATH)
    if len(chunks) != len(emb):
        raise RuntimeError(f"{len(chunks)} chunks but {len(emb)} embeddings")

    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), 500):
        batch = chunks[i : i + 500]
        col.add(
            ids=[c["id"] for c in batch],
            embeddings=emb[i : i + 500].tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[
                {"source": c["source"], "library": c["library"],
                 "section": c["section"], "n_words": c["n_words"]}
                for c in batch
            ],
        )
    return col.count()