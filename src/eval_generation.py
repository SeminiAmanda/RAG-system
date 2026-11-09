"""
Run retrieve -> generate over a questions file and save every answer.
Resumable: questions already present in --out are skipped.

  source scripts/llm_env.sh
  python src/eval_generation.py --questions eval/questions.jsonl --out eval/results/gen_main.jsonl
  python src/eval_generation.py --questions eval/questions_unanswerable.jsonl --out eval/results/gen_unans.jsonl
  python src/eval_generation.py --out eval/results/gen_main.jsonl --summary-only
"""
import argparse
import json
import re
import time
import os
from pathlib import Path

import requests

from generate import generate
from retrieve import Retriever

REFUSAL = "i don't know based on the provided documents"
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def extract_citations(answer: str):
    """Citation numbers like [1], [2][3] or [1, 3]. Code is stripped first so that
    indexing such as dataset[0] is not mistaken for a citation."""
    text = re.sub(r"```.*?```", " ", answer, flags=re.S)   # fenced code blocks
    text = re.sub(r"`[^`\n]*`", " ", text)                 # inline code
    nums = set()
    for group in CITATION_RE.findall(text):
        nums.update(int(n) for n in re.split(r"\s*,\s*", group))
    return sorted(nums)

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


MAX_WAIT = 180  # seconds; a longer rate-limit wait usually means a daily cap


def generate_with_retry(question, passages, tries=6):
    for attempt in range(tries):
        last = attempt == tries - 1
        try:
            return generate(question, passages)
        except requests.HTTPError as e:
            resp = e.response
            status = resp.status_code if resp is not None else None
            if status == 429:
                try:
                    wait = float(resp.headers.get("retry-after", ""))
                except ValueError:
                    wait = 10 * (attempt + 1)
                if wait > MAX_WAIT:
                    detail = resp.text[:300].replace("\n", " ")
                    raise SystemExit(
                        f"\nThe API asks for a {wait:.0f}s wait, which suggests a daily limit.\n"
                        f"Details: {detail}\nProgress is saved; rerun the same command later."
                    )
                if not last:
                    print(f"  rate limited, waiting {wait:.0f}s ...")
                    time.sleep(wait)
                    continue
            elif status in (500, 502, 503, 504) and not last:
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except requests.ConnectionError:
            if not last:
                print("  connection problem, retrying ...")
                time.sleep(5 * (attempt + 1))
                continue
            raise

def is_refusal(answer):
    return REFUSAL in answer.lower().replace("’", "'")


def summarize(out_path):
    recs = read_jsonl(out_path)
    ans = [r for r in recs if r["answerable"]]
    una = [r for r in recs if not r["answerable"]]
    print(f"\n=== {out_path} ===")
    if ans:
        answered = [r for r in ans if not r["refused"]]
        print(f"Answerable questions: {len(ans)}")
        print(f"  wrongly refused         : {len(ans) - len(answered)}")
        if answered:
            print(f"  answers with a citation : {sum(bool(r['cited']) for r in answered)} / {len(answered)}")
            print(f"  all citations valid     : {sum(r['cited_valid'] for r in answered if r['cited'])} / "
                  f"{sum(bool(r['cited']) for r in answered)}")
            print(f"  cites the gold file     : {sum(r['gold_cited'] for r in answered)} / {len(answered)}")
        print(f"  gold file in passages   : {sum(r['gold_retrieved'] for r in ans)} / {len(ans)}")
        for r in ans:
            if r["refused"]:
                print(f"    refused: {r['id']} (top1={r['top1_score']:.2f}) {r['question']}")
    if una:
        refused = [r for r in una if r["refused"]]
        print(f"Unanswerable questions: {len(una)}")
        print(f"  correctly refused       : {len(refused)} / {len(una)}")
        for r in una:
            if not r["refused"]:
                print(f"    ANSWERED anyway: {r['id']} (top1={r['top1_score']:.2f}) {r['question']}")
                print(f"      -> {r['answer'][:200].replace(chr(10), ' ')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-score", type=float, default=None,
                    help="drop passages below this reranker score; refuse without calling the LLM if none remain")
    ap.add_argument("--pause", type=float, default=2.0, help="seconds between LLM calls")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if args.summary_only:
        summarize(out_path)
        return
    if not args.questions:
        raise SystemExit("--questions is required unless --summary-only")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [q for q in read_jsonl(args.questions) if q["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} already done, {len(todo)} to run")

    retriever = Retriever()
    with open(out_path, "a", encoding="utf-8") as out:
        for i, q in enumerate(todo, 1):
            passages = retriever.search(q["question"], k=args.k, mode="hybrid", rerank=True)
            top1 = passages[0]["score"] if passages else -99.0
            if args.min_score is not None:
                passages = [p for p in passages if p["score"] >= args.min_score]

            gated = not passages
            if gated:
                answer = "I don't know based on the provided documents."
            else:
                answer = generate_with_retry(q["question"], passages)
                time.sleep(args.pause)

            cited = extract_citations(answer)
            valid = all(1 <= n <= len(passages) for n in cited)
            cited_sources = [passages[n - 1]["source"] for n in cited if 1 <= n <= len(passages)]
            answerable = q.get("answerable", True)
            raw = q.get("sources") or q.get("source") or []
            gold = {raw} if isinstance(raw, str) else set(raw)

            rec = {
                "id": q["id"],
                "question": q["question"],
                "answerable": answerable,
                "answer": answer,
                "refused": is_refusal(answer),
                "gated": gated,
                "top1_score": top1,
                "cited": cited,
                "cited_valid": valid,
                "cited_sources": cited_sources,
                "gold_retrieved": any(p["source"] in gold for p in passages),
                "gold_cited": any(s in gold for s in cited_sources),
                "passages": [{"source": p["source"], "section": p["section"],
                "prompt_version": os.getenv("PROMPT_VERSION", "1"),              "score": round(p["score"], 2)} for p in passages],
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{i}/{len(todo)}] {q['id']}: {'REFUSED' if rec['refused'] else 'answered'}")

    summarize(out_path)


if __name__ == "__main__":
    main()