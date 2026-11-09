"""
Recompute citation fields from the saved answers (no LLM calls).

  python scripts/rescore.py eval/results/gen_main.jsonl eval/questions.jsonl
Writes <name>_rescored.jsonl and prints the summary.
"""
import json
import sys

sys.path.insert(0, "src")
from eval_generation import extract_citations, summarize  # noqa: E402


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


gen_path, q_path = sys.argv[1], sys.argv[2]

gold = {}
for q in read_jsonl(q_path):
    raw = q.get("sources") or q.get("source") or []
    gold[q["id"]] = {raw} if isinstance(raw, str) else set(raw)

out_path = gen_path.replace(".jsonl", "_rescored.jsonl")
changed = 0
with open(out_path, "w", encoding="utf-8") as f:
    for r in read_jsonl(gen_path):
        cited = extract_citations(r["answer"])
        n = len(r["passages"])
        cited_sources = [r["passages"][k - 1]["source"] for k in cited if 1 <= k <= n]
        if cited != r["cited"]:
            changed += 1
            print(f"  {r['id']}: cited {r['cited']} -> {cited}")
        r.update(
            cited=cited,
            cited_valid=all(1 <= k <= n for k in cited),
            cited_sources=cited_sources,
            gold_cited=any(s in gold.get(r["id"], set()) for s in cited_sources),
        )
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"{changed} record(s) changed -> {out_path}")
summarize(out_path)