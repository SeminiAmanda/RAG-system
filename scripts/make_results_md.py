"""
Build eval/results/RESULTS.md from the saved run logs and generation files.
  python scripts/make_results_md.py
"""
import json
from pathlib import Path

R = Path("eval/results")


def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


runs = {}
for r in read_jsonl(R / "runs.jsonl"):
    runs[r["run"]] = r  # the latest row per run name wins


def hits(r, key="hit@1"):
    n = r["n_answerable"]
    return f"{round(r[key] * n)}/{n}"


out = []

# ---------- 1. retrieval pipeline ----------
out += ["## Retrieval: pipeline stages", "",
        "| method | main hit@1 | main MRR | hard hit@1 | hard hit@10 | hard MRR | latency (ms) |",
        "|---|---|---|---|---|---|---|"]
stages = [("vector", "vector"), ("BM25", "bm25"), ("hybrid", "hybrid"),
          ("vector + rerank", "vector_rerank"), ("hybrid + rerank", "hybrid_rerank")]
for label, key in stages:
    m, h = runs.get(f"{key}_main"), runs.get(f"{key}_hard_v2")
    if m and h:
        out.append(f"| {label} | {hits(m)} | {m['MRR']:.3f} | {hits(h)} | "
                   f"{hits(h, 'hit@10')} | {h['MRR']:.3f} | {m['latency_ms']} |")

# ---------- 2. ablations ----------
out += ["", "## Retrieval: ablations (hit@1 questions, MRR in brackets)", "",
        "| config | vector main | vector hard | hybrid+rerank main | hybrid+rerank hard |",
        "|---|---|---|---|---|"]
configs = []
for name in runs:
    if name.startswith("abl_"):
        c = name.split("_")[1]
        if c not in configs:
            configs.append(c)
for c in configs:
    cells = []
    for mode in ("vector", "hybrid_rerank"):
        for s in ("main", "hard"):
            r = runs.get(f"abl_{c}_{mode}_{s}")
            cells.append(f"{hits(r)} ({r['MRR']:.3f})" if r else "-")
    out.append(f"| {c} | " + " | ".join(cells) + " |")


# ---------- 3. generation ----------
def gen_stats(answerable_files, unans_files):
    ans, una = [], []
    for f in answerable_files:
        ans += [r for r in read_jsonl(R / f) if r["answerable"]]
    for f in unans_files:
        una += [r for r in read_jsonl(R / f) if not r["answerable"]]
    answered = [r for r in ans if not r["refused"]]
    return {
        "answerable": len(ans),
        "wrongly refused": [r["id"] for r in ans if r["refused"]],
        "answered": len(answered),
        "with citation": sum(bool(r["cited"]) for r in answered),
        "cites gold": sum(bool(r["gold_cited"]) for r in answered),
        "unanswerable": len(una),
        "unans refused": sum(r["refused"] for r in una),
    }


def fmt(s):
    return {
        "answerable questions run": str(s["answerable"]),
        "wrongly refused": f"{len(s['wrongly refused'])} {s['wrongly refused']}",
        "answers with a citation": f"{s['with citation']}/{s['answered']}",
        "answers citing the gold file": f"{s['cites gold']}/{s['answered']}",
        "unanswerable refused": f"{s['unans refused']}/{s['unanswerable']}",
    }


v1 = fmt(gen_stats(["gen_main_rescored.jsonl", "gen_hard_v2_rescored.jsonl"],
                   ["gen_main_rescored.jsonl", "gen_unans_v2_rescored.jsonl"]))
v2 = fmt(gen_stats(["gen_main_p2_rescored.jsonl", "gen_hard_v2_p2_rescored.jsonl"],
                   ["gen_main_p2_rescored.jsonl", "gen_unans_v2_p2_rescored.jsonl"]))
out += ["", "## Generation: prompt v1 vs v2", "", "| | prompt v1 | prompt v2 |", "|---|---|---|"]
for k in v1:
    out.append(f"| {k} | {v1[k]} | {v2[k]} |")

text = "\n".join(out) + "\n"
(R / "RESULTS.md").write_text(text, encoding="utf-8")
print(text)