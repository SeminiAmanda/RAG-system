import json
import time

import chromadb
from sentence_transformers import SentenceTransformer

from config import CHUNKS_PATH, EMBED_MODEL, EMBED_WITH_HEADING, INDEX_DIR, collection_name


def main():
    chunks = [json.loads(line) for line in open(CHUNKS_PATH, encoding="utf-8")]
    print(f"Loaded {len(chunks)} chunks")

    if EMBED_WITH_HEADING:
        to_embed = [f"{c['section']}\n{c['text']}" for c in chunks]
    else:
        to_embed = [c["text"] for c in chunks]

    model = SentenceTransformer(EMBED_MODEL, device="cpu")
    print(f"Model: {EMBED_MODEL} (max_seq_length={model.max_seq_length})")

    t0 = time.time()
    embeddings = model.encode(
        to_embed, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    )
    print(f"Embedded in {time.time() - t0:.0f}s, shape={embeddings.shape}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(INDEX_DIR))
    name = collection_name()
    try:
        client.delete_collection(name)
    except Exception:
        pass
    col = client.create_collection(name, metadata={"hnsw:space": "cosine"})

    B = 500
    for i in range(0, len(chunks), B):
        batch = chunks[i : i + B]
        col.add(
            ids=[c["id"] for c in batch],
            embeddings=embeddings[i : i + B].tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "source": c["source"],
                    "library": c["library"],
                    "section": c["section"],
                    "n_words": c["n_words"],
                }
                for c in batch
            ],
        )
    print(f"Indexed {col.count()} chunks into '{name}' at {INDEX_DIR}")


if __name__ == "__main__":
    main()