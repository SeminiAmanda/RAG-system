import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "eval/questions_unanswerable.jsonl"

chunks = [json.loads(l) for l in open("data/processed/chunks.jsonl", encoding="utf-8")]
texts = [(c["source"], c["section"], (c["section"] + "\n" + c["text"]).lower()) for c in chunks]

for line in open(path, encoding="utf-8"):
    if not line.strip():
        continue
    q = json.loads(line)
    terms = [t.lower() for t in q.get("terms", [])]
    hits = [(s, sec) for s, sec, t in texts if terms and all(term in t for term in terms)]
    status = "OK" if not hits else f"REVIEW ({len(hits)} chunk(s) contain the terms)"
    print(f"{q['id']}: {status}   terms={terms}")
    for s, sec in hits[:3]:
        print(f"     {s} | {sec}")