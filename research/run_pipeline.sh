#!/usr/bin/env bash
# End-to-end driver. Run from project root after data/sample.jsonl exists.
#
# Two providers run concurrently — Anthropic and Google have independent
# rate limits — so wall-clock for the LLM stages is roughly halved.
#
# Stability runs (multiple temperature seeds) are intentionally OMITTED from
# the default sweep to keep wall-clock reasonable; flip ENABLE_STABILITY=1
# below if you want them.
set -uo pipefail
cd "$(dirname "$0")"
PY=".venv/bin/python"
ENABLE_STABILITY="${ENABLE_STABILITY:-0}"

mkdir -p outputs/raw eval

run_pair_parallel() {
  local run_id="$1"
  local temp="$2"
  shift 2
  local prompts=("$@")
  echo "[pipeline] launching parallel ($run_id @ T=$temp): claude_haiku + gemini_flash, prompts ${prompts[*]}"
  $PY src/run_models.py --models claude_haiku_4_5 --prompts "${prompts[@]}" \
      --temperature "$temp" --run-id "$run_id" \
      > "outputs/raw/_log_${run_id}_haiku.txt" 2>&1 &
  pid_h=$!
  $PY src/run_models.py --models gemini_2_5_flash --prompts "${prompts[@]}" \
      --temperature "$temp" --run-id "$run_id" \
      > "outputs/raw/_log_${run_id}_gemini.txt" 2>&1 &
  pid_g=$!
  wait $pid_h; rc_h=$?
  wait $pid_g; rc_g=$?
  echo "[pipeline] $run_id done. haiku rc=$rc_h gemini rc=$rc_g"
  if [ $rc_h -ne 0 ] || [ $rc_g -ne 0 ]; then
    echo "[pipeline] FAILED — see outputs/raw/_log_${run_id}_*.txt"
    return 1
  fi
}

echo "=== run_models: PRIMARY (T=0, prompts A & B) ==="
run_pair_parallel primary 0.0 A B || exit 1

if [ "$ENABLE_STABILITY" = "1" ]; then
  echo "=== run_models: STABILITY (T=0.7, 3 seeds, prompt A only) ==="
  for seed in 1 2 3; do
    run_pair_parallel "stability_$seed" 0.7 A || exit 1
  done
else
  echo "=== stability runs SKIPPED (set ENABLE_STABILITY=1 to enable) ==="
fi

echo "=== reliability stack ==="
$PY src/cluster_themes.py        || echo "cluster_themes: failed (continuing)"
$PY src/grounding.py             || echo "grounding: failed (continuing)"
$PY src/literature_anchor.py     || echo "literature_anchor: failed (continuing)"
$PY src/llm_judge.py             || echo "llm_judge: failed (continuing)"
$PY src/reliability_score.py     || echo "reliability_score: failed (continuing)"

echo "=== build_report ==="
$PY src/build_report.py

echo "=== done. tables in manuscript/tables/, figures in manuscript/figures/ ==="
