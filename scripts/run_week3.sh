#!/usr/bin/env bash
set -e

for pair in "eval/questions.jsonl:main" "eval/questions_hard.jsonl:hard"; do
  Q="${pair%%:*}"
  N="${pair##*:}"
  python src/evaluate.py --run "vector_$N"                 --questions "$Q" --mode vector
  python src/evaluate.py --run "bm25_$N"                   --questions "$Q" --mode bm25
  python src/evaluate.py --run "hybrid_$N"                 --questions "$Q" --mode hybrid
  python src/evaluate.py --run "vector_rerank_$N"          --questions "$Q" --mode vector --rerank
  python src/evaluate.py --run "hybrid_rerank_$N"          --questions "$Q" --mode hybrid --rerank
  python src/evaluate.py --run "hybrid_rerank_dedupe_$N"   --questions "$Q" --mode hybrid --rerank --pool 30 --max-per-source 2
done