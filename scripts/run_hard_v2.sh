#!/usr/bin/env bash
set -e
Q=eval/questions_hard_v2.jsonl
python src/evaluate.py --run vector_hard_v2        --questions $Q --mode vector
python src/evaluate.py --run bm25_hard_v2          --questions $Q --mode bm25
python src/evaluate.py --run hybrid_hard_v2        --questions $Q --mode hybrid
python src/evaluate.py --run vector_rerank_hard_v2 --questions $Q --mode vector --rerank
python src/evaluate.py --run hybrid_rerank_hard_v2 --questions $Q --mode hybrid --rerank