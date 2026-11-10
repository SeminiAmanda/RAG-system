import json

rows = [json.loads(l) for l in open("eval/results/runs.jsonl", encoding="utf-8") if l.strip()]
latest = {r["run"]: r for r in rows}

configs = []
for name in latest:
    if name.startswith("abl_"):
        cfg = name.split("_")[1]
        if cfg not in configs:
            configs.append(cfg)

cols = [("vector", "main"), ("vector", "hard"), ("hybrid_rerank", "main"), ("hybrid_rerank", "hard")]
head = ["config"]
for mode, s in cols:
    head += [f"{mode} {s} hit@1", "MRR"]
print("| " + " | ".join(head) + " |")
print("|" + "---|" * len(head))

for cfg in configs:
    cells = [cfg]
    for mode, s in cols:
        r = latest.get(f"abl_{cfg}_{mode}_{s}")
        cells += [f"{r['hit@1']:.3f}", f"{r['MRR']:.3f}"] if r else ["-", "-"]
    print("| " + " | ".join(cells) + " |")