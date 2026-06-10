#!/usr/bin/env bash
# Run sample + full pipeline end-to-end. Touches data/.pipeline_done at the end.
# Collection is assumed to have been done already (data/raw/ has JSONL files).

set -uo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"

LOG="data/orchestrate.log"
exec >> "$LOG" 2>&1

echo
echo "[orchestrate] $(date -u) starting orchestrator (collection assumed complete)"
echo "[orchestrate] data/raw contents:"
ls -la data/raw/

if [ ! -f data/sample.jsonl ]; then
  echo "[orchestrate] $(date -u) running sample_corpus.py"
  $PY src/sample_corpus.py
  echo "[orchestrate] $(date -u) sample_corpus done"
else
  echo "[orchestrate] data/sample.jsonl already exists, skipping sample_corpus.py"
fi

echo "[orchestrate] $(date -u) launching run_pipeline.sh"
bash run_pipeline.sh

echo "[orchestrate] $(date -u) pipeline complete"
date -u > data/.pipeline_done
echo "[orchestrate] $(date -u) wrote data/.pipeline_done"
