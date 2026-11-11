"""
Export the exact embeddings from the local Chroma index so a deployment can rebuild
the index in seconds, without running the embedding model over the whole corpus.

  python scripts/export_embeddings.py
"""
import json
import sys

import chromadb
import numpy as np

sys.path.insert(0, "src")
from config import CHUNKS_PATH, INDEX_DIR, collection_name  # noqa: E402

with open(CHUNKS_PATH, encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]

col = chromadb.PersistentClient(path=str(INDEX_DIR)).get_collection(collection_name())
res = col.get(include=["embeddings"])
by_id = dict(zip(res["ids"], res["embeddings"]))

missing = [c["id"] for c in chunks if c["id"] not in by_id]
assert not missing, f"{len(missing)} chunks have no embedding, e.g. {missing[:3]}"

emb = np.array([by_id[c["id"]] for c in chunks], dtype=np.float32)
np.save("corpus/embeddings.npy", emb)
print(f"saved {emb.shape} to corpus/embeddings.npy ({emb.nbytes / 1e6:.1f} MB)")