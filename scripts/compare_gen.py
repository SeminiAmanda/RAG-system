"""
Compare two generation runs on the questions both have answered.
  python scripts/compare_gen.py eval/results/gen_hard_v2_rescored.jsonl eval/results/gen_hard_v2_p2_rescored.jsonl
"""
import json
import sys


def read(path):
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}


a, b = read(sys.argv[1]), read(sys.argv[2])
ids = sorted(set(a) & set(b))
print(f"{len(ids)} questions in common ({len(a)} vs {len(b)} in each file)")


def stats(recs):
    ans = [r for r in recs if r["answerable"]]
    una = [r for r in recs if not r["answerable"]]
    answered = [r for r in ans if not r["refused"]]
    return {
        "answerable": len(ans),
        "wrongly refused": sum(r["refused"] for r in ans),
        "answered, no citation": sum(not r["cited"] for r in answered),
        "cites the gold file": sum(r["gold_cited"] for r in answered),
        "unanswerable": len(una),
        "unanswerable refused": sum(r["refused"] for r in una),
        "avg answer words": round(sum(len(r["answer"].split()) for r in answered) / max(len(answered), 1)),
    }


sa, sb = stats([a[i] for i in ids]), stats([b[i] for i in ids])
print(f"\n{'':<26}{'run A':>8}{'run B':>8}")
for k in sa:
    print(f"{k:<26}{sa[k]:>8}{sb[k]:>8}")

print("\nQuestions where refusal differs:")
for i in ids:
    if a[i]["refused"] != b[i]["refused"]:
        print(f"  {i}: A refused={a[i]['refused']}  B refused={b[i]['refused']}  {a[i]['question']}")