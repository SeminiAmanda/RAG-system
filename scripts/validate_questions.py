"""
Validate an evaluation questions file against data/processed/chunks.jsonl.

Usage:
  python scripts/validate_questions.py                              # eval/questions.jsonl
  python scripts/validate_questions.py eval/questions_hard_v2.jsonl
"""
import json
import sys

CHUNKS_PATH = "data/processed/chunks.jsonl"
DEFAULT_QUESTIONS = "eval/questions.jsonl"

path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTIONS

with open(CHUNKS_PATH, encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f if line.strip()]
sources = {c["source"] for c in chunks}
pairs = {(c["source"], c["section"]) for c in chunks}

questions = []
with open(path, encoding="utf-8") as f:
    for lineno, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            questions.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"BAD JSON on line {lineno}: {e}")
            sys.exit(1)

problems = 0
seen_ids = set()

for q in questions:
    qid = q.get("id", "<no id>")

    for field in ("id", "question", "answerable"):
        if field not in q:
            print(f"MISSING FIELD '{field}': {qid}")
            problems += 1

    if qid in seen_ids:
        print(f"DUPLICATE ID: {qid}")
        problems += 1
    seen_ids.add(qid)

    if not q.get("answerable", True):
        continue

    # 'source' must be a string; lists belong in 'sources'
    if "source" in q and not isinstance(q["source"], str):
        print(f"BAD TYPE: {qid}: 'source' must be a string; put lists in 'sources'")
        problems += 1
        continue
    srcs = q.get("sources")
    if srcs is not None and (
        not isinstance(srcs, list) or not all(isinstance(s, str) for s in srcs)
    ):
        print(f"BAD TYPE: {qid}: 'sources' must be a flat list of strings")
        problems += 1
        continue

    gold = srcs or ([q["source"]] if q.get("source") else [])
    if not gold:
        print(f"NO SOURCE: {qid}")
        problems += 1
        continue

    missing = [s for s in gold if s not in sources]
    if missing:
        print(f"MISSING SOURCE : {qid} {missing}")
        problems += 1
        continue

    section = q.get("section")
    if section and not any((s, section) in pairs for s in gold):
        print(f"MISSING SECTION: {qid} {section!r}")
        problems += 1

answerable = sum(1 for q in questions if q.get("answerable", True))
print(
    f"{path}: {len(questions)} questions "
    f"({answerable} answerable, {len(questions) - answerable} unanswerable), "
    f"{problems} problems"
)
sys.exit(1 if problems else 0)