import json
import sys
from collections import defaultdict

chunks = [json.loads(l) for l in open("data/processed/chunks.jsonl", encoding="utf-8")]
for term in sys.argv[1:]:
    by_source = defaultdict(list)
    for c in chunks:
        if term in c["text"]:
            by_source[c["source"]].append(c["section"])
    print(f"\n{term!r}: {len(by_source)} file(s)")
    for src, secs in sorted(by_source.items()):
        print(f"  {src}  ({len(secs)} chunk(s))  e.g. {secs[0]}")