import json
import sys

rows = [json.loads(l) for l in open("eval/results/runs.jsonl", encoding="utf-8") if l.strip()]
latest = {}
for r in rows:           # keep the most recent row per run name
    latest[r["run"]] = r

needle = sys.argv[1] if len(sys.argv) > 1 else ""
cols = ["hit@1", "hit@3", "hit@5", "hit@10", "MRR", "section_hit@5", "latency_ms"]
print("| run | n | " + " | ".join(cols) + " |")
print("|" + "---|" * (len(cols) + 2))
for name, r in latest.items():
    if needle in name:
        print(f"| {name} | {r.get('n_answerable', '')} | " + " | ".join(str(r.get(c, "")) for c in cols) + " |")