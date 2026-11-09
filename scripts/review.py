"""
Review helper for generated answers (output of src/eval_generation.py).

  python scripts/review.py issues eval/results/gen_main.jsonl eval/questions.jsonl
  python scripts/review.py show   eval/results/gen_unans.jsonl u02 u15
  python scripts/review.py sheet  eval/results/gen_main.jsonl --n 15 --out eval/results/review_main.md
  python scripts/review.py tally  eval/results/review_main.md
"""
import argparse
import json
import random
import re
import sys

sys.path.insert(0, "src")  # so `retrieve` and `config` can be imported


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_passages(retriever, rec, k=5):
    """Re-run retrieval with the same settings as eval_generation.py to read the passages."""
    passages = retriever.search(rec["question"], k=k, mode="hybrid", rerank=True)
    if [p["source"] for p in passages] != [p["source"] for p in rec["passages"]]:
        print(f"  (warning: passages for {rec['id']} differ from the stored run)")
    return passages


def print_record(rec, passages):
    cited = rec["cited"]
    print("=" * 78)
    print(f"{rec['id']}  answerable={rec['answerable']}  refused={rec['refused']}  top1={rec['top1_score']:.2f}")
    print(f"Q: {rec['question']}\n\nANSWER:\n{rec['answer']}\n\ncited: {cited}")
    for n, p in enumerate(passages, 1):
        limit = 900 if n in cited else 250
        mark = "   <== cited" if n in cited else ""
        print(f"\n--- [{n}] {p['source']} | {p['section']} (score {p['score']:.2f}){mark}")
        print(p["text"][:limit])


def cmd_issues(gen_path, q_path):
    gold = {}
    for q in read_jsonl(q_path):
        gold[q["id"]] = set(q.get("sources") or ([q["source"]] if q.get("source") else []))
    recs = read_jsonl(gen_path)
    checks = [
        ("Answered WITHOUT any citation",
         lambda r: r["answerable"] and not r["refused"] and not r["cited"]),
        ("Citation number out of range",
         lambda r: bool(r["cited"]) and not r["cited_valid"]),
        ("Cited passages do not include the gold file",
         lambda r: r["answerable"] and not r["refused"] and bool(r["cited"]) and not r["gold_cited"]),
        ("Unanswerable question but ANSWERED",
         lambda r: not r["answerable"] and not r["refused"]),
    ]
    for title, cond in checks:
        hits = [r for r in recs if cond(r)]
        print(f"\n{title}: {len(hits)}")
        for r in hits:
            print(f"  {r['id']}: {r['question']}")
            print(f"     cited={r['cited']} cited_sources={sorted(set(r['cited_sources']))} "
                  f"gold={sorted(gold.get(r['id'], []))}")


def cmd_show(gen_path, ids):
    from retrieve import Retriever
    retriever = Retriever()
    recs = {r["id"]: r for r in read_jsonl(gen_path)}
    for i in ids:
        print_record(recs[i], get_passages(retriever, recs[i]))


def cmd_sheet(gen_path, n, out_path, seed):
    from retrieve import Retriever
    retriever = Retriever()
    recs = read_jsonl(gen_path)
    pool = [r for r in recs if r["answerable"] and not r["refused"]]
    random.Random(seed).shuffle(pool)
    chosen = pool[:n] + [r for r in recs if not r["answerable"] and not r["refused"]]

    lines = [
        "# Answer review", "",
        "For each item replace the `?` after VERDICT with:",
        "2 = correct and supported by the passages, 1 = partly correct or partly unsupported, "
        "0 = wrong or not supported by the passages.", "",
    ]
    for rec in chosen:
        passages = get_passages(retriever, rec)
        lines += [f"## {rec['id']}", f"**Question:** {rec['question']}", "", "**Answer:**"]
        lines += ["> " + line for line in rec["answer"].splitlines()]
        shown = rec["cited"] or [1, 2]
        lines += ["", "**Passages cited**" + ("" if rec["cited"] else " (none cited, showing top 2)") + ":"]
        for k in shown:
            if 1 <= k <= len(passages):
                p = passages[k - 1]
                lines += [f"### [{k}] {p['source']} | {p['section']}", "```text", p["text"][:1200], "```"]
        lines += ["", "VERDICT: ?", "NOTES:", "", "---", ""]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(chosen)} items to {out_path}")


def cmd_tally(path):
    text = open(path, encoding="utf-8").read()
    marks = re.findall(r"^VERDICT:\s*(\S+)", text, flags=re.M)
    done = [m for m in marks if m in {"0", "1", "2"}]
    print(f"{len(marks)} items, {len(done)} scored")
    for v in "210":
        if done:
            print(f"  verdict {v}: {done.count(v)} ({done.count(v) / len(done):.0%})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("issues"); p.add_argument("gen"); p.add_argument("questions")
    p = sub.add_parser("show"); p.add_argument("gen"); p.add_argument("ids", nargs="+")
    p = sub.add_parser("sheet"); p.add_argument("gen"); p.add_argument("--n", type=int, default=15)
    p.add_argument("--out", required=True); p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("tally"); p.add_argument("path")
    a = ap.parse_args()

    if a.cmd == "issues":
        cmd_issues(a.gen, a.questions)
    elif a.cmd == "show":
        cmd_show(a.gen, a.ids)
    elif a.cmd == "sheet":
        cmd_sheet(a.gen, a.n, a.out, a.seed)
    elif a.cmd == "tally":
        cmd_tally(a.path)


if __name__ == "__main__":
    main()