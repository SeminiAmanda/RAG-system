"""
Retriever with several modes:
  vector : dense similarity search (Chroma)
  bm25   : keyword search (rank_bm25)
  hybrid : vector + bm25 fused with Reciprocal Rank Fusion (RRF)
Optional extras: cross-encoder reranking, per-source limit, library filter.

Scores are NOT comparable across modes (cosine similarity, BM25 score,
RRF score and reranker probability all have different scales).
"""
import json
import re

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (
    CHUNKS_PATH,
    EMBED_MODEL,
    EMBED_WITH_HEADING,
    INDEX_DIR,
    QUERY_PREFIXES,
    RERANK_MODEL,
    RRF_K,
    collection_name,
)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "to", "of", "in", "on", "for", "and", "or",
    "do", "does", "i", "my", "me", "can", "how", "what", "which", "with", "it", "this",
    "that", "be", "by", "as", "at", "from", "if", "so", "you", "your", "we", "should",
}
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str):
    """Lowercase word tokens. snake_case identifiers are kept whole AND split into
    parts, so 'weight_sync_steps' matches the identifier and the words in it."""
    tokens = []
    for tok in TOKEN_RE.findall(text):
        low = tok.lower()
        if low in STOPWORDS:
            continue
        tokens.append(low)
        parts = low.split("_")
        if len(parts) > 1:
            tokens.extend(p for p in parts if p and p not in STOPWORDS)
    return tokens


class Retriever:
    def __init__(self, model_name: str = EMBED_MODEL):
        # dense side
        self.model = SentenceTransformer(model_name, device="cpu")
        self.prefix = QUERY_PREFIXES.get(model_name, "")
        client = chromadb.PersistentClient(path=str(INDEX_DIR))
        self.col = client.get_collection(collection_name(model_name))

        # chunk store (single source of truth for text and metadata)
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f if line.strip()]
        self.by_id = {c["id"]: c for c in self.chunks}
        self.libraries = np.array([c["library"] for c in self.chunks])

        # keyword side (same text that was embedded)
        docs = [
            f"{c['section']}\n{c['text']}" if EMBED_WITH_HEADING else c["text"]
            for c in self.chunks
        ]
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])

        self._reranker = None  # loaded lazily, only if rerank=True

    # ---------------- first-stage retrievers ----------------
    def _vector(self, query, n, library=None):
        emb = self.model.encode([self.prefix + query], normalize_embeddings=True)
        kwargs = {"where": {"library": library}} if library else {}
        res = self.col.query(query_embeddings=emb.tolist(), n_results=n, **kwargs)
        return [(cid, 1 - dist) for cid, dist in zip(res["ids"][0], res["distances"][0])]

    def _bm25(self, query, n, library=None):
        scores = self.bm25.get_scores(tokenize(query))
        if library:
            scores = np.where(self.libraries == library, scores, 0.0)
        top = np.argsort(-scores)[:n]
        return [(self.chunks[i]["id"], float(scores[i])) for i in top if scores[i] > 0]

    @staticmethod
    def _rrf(ranked_lists, k=RRF_K):
        """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank)."""
        fused = {}
        for lst in ranked_lists:
            for rank, (cid, _) in enumerate(lst, 1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    # ---------------- helpers ----------------
    def _record(self, cid, score):
        c = self.by_id[cid]
        return {
            "id": cid,
            "text": c["text"],
            "source": c["source"],
            "library": c["library"],
            "section": c["section"],
            "score": float(score),
        }

    def _rerank(self, query, cands):
        if self._reranker is None:
            self._reranker = CrossEncoder(RERANK_MODEL, device="cpu", max_length=512)
        pairs = [(query, f"{c['section']}\n{c['text']}") for c in cands]
        scores = self._reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        ranked = sorted(zip(cands, scores), key=lambda x: x[1], reverse=True)
        return [{**c, "score": float(s)} for c, s in ranked]

    @staticmethod
    def _limit_per_source(cands, max_per_source):
        counts, out = {}, []
        for c in cands:
            counts[c["source"]] = counts.get(c["source"], 0) + 1
            if counts[c["source"]] <= max_per_source:
                out.append(c)
        return out

    # ---------------- public API ----------------
    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "vector",
        rerank: bool = False,
        pool: int = 20,
        max_per_source: int | None = None,
        library: str | None = None,
    ):
        """
        mode            'vector' | 'bm25' | 'hybrid'
        rerank          rerank the top `pool` candidates with a cross-encoder
        pool            how many first-stage candidates the reranker sees
        max_per_source  keep at most this many chunks from any one file
        library         restrict to one library, e.g. 'peft'
        """
        n = max(pool, k * 4)
        if mode == "vector":
            ranked = self._vector(query, n, library)
        elif mode == "bm25":
            ranked = self._bm25(query, n, library)
        elif mode == "hybrid":
            ranked = self._rrf(
                [self._vector(query, n, library), self._bm25(query, n, library)]
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

        cands = [self._record(cid, s) for cid, s in ranked]
        if rerank:
            cands = self._rerank(query, cands[:pool])
        if max_per_source:
            cands = self._limit_per_source(cands, max_per_source)
        return cands[:k]


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "weight_sync_steps"
    r = Retriever()
    for mode, rr in [("vector", False), ("bm25", False), ("hybrid", False), ("hybrid", True)]:
        print(f"\n== {mode}{' + rerank' if rr else ''} ==")
        for i, x in enumerate(r.search(q, k=5, mode=mode, rerank=rr), 1):
            print(f"{i}. [{x['score']:.3f}] {x['source']} | {x['section']}")