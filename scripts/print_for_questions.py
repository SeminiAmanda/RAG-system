import json
import random

chunks = [json.loads(l) for l in open("data/processed/chunks.jsonl", encoding="utf-8")]
random.seed(7)
for c in random.sample(chunks, 40):
    print("=" * 70)
    print(c["id"], "|", c["source"], "|", c["section"])
    print(c["text"][:500])
    