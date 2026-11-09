"""
Evaluate retrieval quality.

Metrics (on answerable questions):
  hit@k          fraction of questions where a correct source file appears in the top k
  MRR            mean of 1/rank of the first correct source (0 if not in top --k)
  section_hit@5  correct source AND exact section appear in the top 5

Usage:
  python src/evaluate.py --run vector_main
  python src/evaluate.py --run hybrid_rerank_hard --questions eval/questions_hard.jsonl \
      --mode hybrid --rerank
"""
import argparse
import json
import time
from pathlib import Path

from config import EMBED_MODEL, EMBED_WITH_HEADING, QUESTIONS_PATH
from retrieve import Retriever

KS = [1, 3, 5, 10]
OUT_DIR = Path("eval/results")


def load_questions(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def gold_sources(q):
    """Accepts 'sources' (list) or 'source' (string); tolerates a list in either key."""
    raw = q.get("sources") or q["source"]
    return {raw} if isinstance(raw, str) else set(raw)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="baseline", help="name for this experiment")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--questions", default=str(QUESTIONS_PATH))
    ap.add_argument("--mode", default="vector", choices=["vector", "bm25", "hybrid"])
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--pool", type=int, default=20, help="candidates the reranker sees")
    ap.add_argument("--max-per-source", type=int, default=None)
    args = ap.parse_args()

    if args.k < max(KS):
        raise SystemExit(f"--k must be at least {max(KS)}")

    questions = load_questions(args.questions)
    answerable = [q for q in questions if q.get("answerable", True)]
    unanswerable = [q for q in questions if not q.get("answerable", True)]
    if not answerable:
        raise SystemExit("No answerable questions found in the file.")

    retriever = Retriever()

    def run_search(text, k):
        return retriever.search(
            text,
            k=k,
            mode=args.mode,
            rerank=args.rerank,
            pool=args.pool,
            max_per_source=args.max_per_source,
        )

    hits = {k: 0 for k in KS}
    section_hit5 = 0
    rr_sum = 0.0
    top1_answerable = []
    misses, weak = [], []

    t0 = time.time()
    for q in answerable:
        results = run_search(q["question"], args.k)
        gold = gold_sources(q)
        top1_answerable.append(results[0]["score"] if results else 0.0)

        rank = next((i + 1 for i, r in enumerate(results) if r["source"] in gold), None)
        got = [r["source"] for r in results[:3]]

        if rank is None:
            misses.append({"id": q["id"], "question": q["question"], "gold": sorted(gold), "got": got})
        else:
            rr_sum += 1 / rank
            for k in KS:
                if rank <= k:
                    hits[k] += 1
            if rank != 1:
                weak.append(
                    {"id": q["id"], "question": q["question"], "gold": sorted(gold), "rank": rank, "got": got}
                )

        if q.get("section") and any(
            r["source"] in gold and r["section"] == q["section"] for r in results[:5]
        ):
            section_hit5 += 1

    latency_ms = (time.time() - t0) / len(answerable) * 1000

    top1_unanswerable = []
    for q in unanswerable:
        res = run_search(q["question"], 1)
        top1_unanswerable.append(res[0]["score"] if res else 0.0)

    n = len(answerable)
    metrics = {f"hit@{k}": round(hits[k] / n, 3) for k in KS}
    metrics["MRR"] = round(rr_sum / n, 3)
    metrics["section_hit@5"] = round(section_hit5 / n, 3)

    # ---------------- report ----------------
    config_str = (
        f"mode={args.mode} rerank={args.rerank} pool={args.pool} "
        f"max_per_source={args.max_per_source}"
    )
    print(f"\nRun: {args.run}")
    print(f"Questions file: {args.questions}")
    print(f"Embedding model: {EMBED_MODEL}   heading embedded: {EMBED_WITH_HEADING}")
    print(f"Config: {config_str}")
    print(f"Answerable questions: {n}   avg latency: {latency_ms:.0f} ms/query\n")
    for name, value in metrics.items():
        print(f"  {name:<14}{value:.3f}")

    print("\n(top-1 scores below are on the scale of the chosen mode)")
    print(f"Top-1 score, answerable   : mean {sum(top1_answerable) / len(top1_answerable):.3f}")
    if top1_unanswerable:
        print(
            f"Top-1 score, unanswerable : mean "
            f"{sum(top1_unanswerable) / len(top1_unanswerable):.3f}  (n={len(unanswerable)})"
        )

    print(f"\nMisses (correct source not in top {args.k}): {len(misses)}")
    for m in misses:
        print(f"  {m['id']}: {m['question']}\n     gold={m['gold']}\n     got ={m['got']}")

    print(f"\nFound but not at rank 1: {len(weak)}")
    for w in weak:
        print(f"  {w['id']} (rank {w['rank']}): {w['question']}\n     gold={w['gold']}\n     got ={w['got']}")

    # ---------------- save ----------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "run": args.run,
        "questions": args.questions,
        "model": EMBED_MODEL,
        "embed_with_heading": EMBED_WITH_HEADING,
        "mode": args.mode,
        "rerank": args.rerank,
        "pool": args.pool,
        "max_per_source": args.max_per_source,
        "n_answerable": n,
        "latency_ms": round(latency_ms),
        **metrics,
    }
    with open(OUT_DIR / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    (OUT_DIR / f"{args.run}_misses.json").write_text(
        json.dumps(misses, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT_DIR / f"{args.run}_weak.json").write_text(
        json.dumps(weak, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved to {OUT_DIR}: runs.jsonl (appended), {args.run}_misses.json, {args.run}_weak.json")


if __name__ == "__main__":
    main()