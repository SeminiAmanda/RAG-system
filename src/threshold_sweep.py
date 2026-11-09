"""
For each threshold T: refuse when the top-1 reranker score < T.
  wrongly refused  = answerable questions that would be refused (bad)
  correctly refused = unanswerable questions that would be refused (good)
"""
import json
import statistics
import sys
from pathlib import Path

from retrieve import Retriever

FILES = sys.argv[1:] or [
    "eval/questions.jsonl",
    "eval/questions_hard_v2.jsonl",
    "eval/questions_unanswerable.jsonl",
]

questions = []
for p in FILES:
    with open(p, encoding="utf-8") as f:
        questions += [json.loads(line) for line in f if line.strip()]

retriever = Retriever()
rows = []
for q in questions:
    res = retriever.search(q["question"], k=1, mode="hybrid", rerank=True)
    score = res[0]["score"] if res else -99.0
    rows.append({"id": q["id"], "answerable": q.get("answerable", True),
                 "score": score, "question": q["question"]})

ans = sorted(r["score"] for r in rows if r["answerable"])
una = sorted(r["score"] for r in rows if not r["answerable"])


def describe(name, xs):
    print(f"{name:<13} n={len(xs):<3} min={xs[0]:6.2f}  median={statistics.median(xs):6.2f}  max={xs[-1]:6.2f}")


print()
describe("answerable", ans)
describe("unanswerable", una)

print("\n threshold | wrongly refused (answerable) | correctly refused (unanswerable)")
print("-----------|------------------------------|---------------------------------")
t = -6.0
while t <= 6.0:
    wrong = sum(s < t for s in ans)
    right = sum(s < t for s in una)
    print(f"  {t:6.1f}   |  {wrong:3d} / {len(ans)} = {wrong / len(ans):5.1%}"
          f"          |  {right:3d} / {len(una)} = {right / len(una):5.1%}")
    t += 1.0

print("\nLowest-scoring ANSWERABLE questions (would be refused first):")
for r in sorted((r for r in rows if r["answerable"]), key=lambda r: r["score"])[:6]:
    print(f"  {r['score']:6.2f}  {r['id']}: {r['question']}")

print("\nHighest-scoring UNANSWERABLE questions (would slip through):")
for r in sorted((r for r in rows if not r["answerable"]), key=lambda r: -r["score"])[:6]:
    print(f"  {r['score']:6.2f}  {r['id']}: {r['question']}")

Path("eval/results").mkdir(parents=True, exist_ok=True)
Path("eval/results/top1_scores.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")