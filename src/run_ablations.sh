#!/usr/bin/env bash
# Usage: bash scripts/run_ablations.sh            (runs base c100 c400 nohead minilm)
#        bash scripts/run_ablations.sh bgebase    (runs only the listed configs)
set -e
set -o pipefail
LOG="eval/results/ablation_log.txt"
mkdir -p eval/results

run_config () {
  NAME="$1"; MODEL="$2"; TARGET="$3"; HEAD="${4:-1}"
  export EMBED_MODEL="$MODEL" TARGET_WORDS="$TARGET" EMBED_WITH_HEADING="$HEAD"

  if [ "$TARGET" = "200" ]; then
    unset CHUNK_TAG
    export CHUNKS_PATH="data/processed/chunks.jsonl"
  else
    export CHUNK_TAG="c${TARGET}"
    export CHUNKS_PATH="data/processed/chunks_c${TARGET}.jsonl"
    [ -f "$CHUNKS_PATH" ] || python src/ingest.py | tee -a "$LOG"
  fi

  echo "=== $NAME: $MODEL, ${TARGET}-word chunks, heading=$HEAD ===" | tee -a "$LOG"
  if [ "$NAME" != "base" ]; then   # the baseline index already exists
    python src/embed.py 2>&1 | grep -E "Loaded|Model:|Embedded|Indexed" | tee -a "$LOG"
  fi

  for SET in "eval/questions.jsonl:main" "eval/questions_hard_v2.jsonl:hard"; do
    Q="${SET%%:*}"; S="${SET##*:}"
    python src/evaluate.py --run "abl_${NAME}_vector_${S}" --questions "$Q" --mode vector
    python src/evaluate.py --run "abl_${NAME}_hybrid_rerank_${S}" --questions "$Q" --mode hybrid --rerank
  done
}

CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then CONFIGS=(base c100 c400 nohead minilm); fi

for c in "${CONFIGS[@]}"; do
  case "$c" in
    base)    run_config base    BAAI/bge-small-en-v1.5                    200 1 ;;
    c100)    run_config c100    BAAI/bge-small-en-v1.5                    100 1 ;;
    c400)    run_config c400    BAAI/bge-small-en-v1.5                    400 1 ;;
    nohead)  run_config nohead  BAAI/bge-small-en-v1.5                    200 0 ;;
    minilm)  run_config minilm  sentence-transformers/all-MiniLM-L6-v2    200 1 ;;
    bgebase) run_config bgebase BAAI/bge-base-en-v1.5                     200 1 ;;
    *) echo "unknown config: $c"; exit 1 ;;
  esac
done
echo "done"
