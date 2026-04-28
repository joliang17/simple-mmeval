#!/usr/bin/env bash

set -euo pipefail

# Example script: run scoring (exact -> template -> llm) on evalkit outputs.
# Defaults are intentionally conservative for cost/time.
#
# Usage:
#   OPENAI_API_KEY=xxx bash scripts/examples/run_score_evalkit_llm.sh
#
# Optional env vars:
#   OUT_DIR=work_dirs/evalkit_all_qwen
#   SCORE_RESULT_GLOB='VStarBench/result.json'   # or '*/result.json'
#   SCORE_OUTPUT_NAME=score_llm.json
#   PARALLEL_PER_TASK=1   # sample workers inside each result.json
#   MATCHING_ORDER='exact,template,llm'
#   JUDGE_PROVIDER='openai'
#   JUDGE_MODEL='gpt-5'
#   JUDGE_INCLUDE_REASON='false'
#   SCORE_RESUME='true'
#   SCORE_SAVE_FREQ=20

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="./:${PYTHONPATH:-}"

OUT_DIR="${OUT_DIR:-work_dirs/evalkit_all_qwen}"
SCORE_RESULT_GLOB="${SCORE_RESULT_GLOB:-VStarBench/result.json}"
SCORE_OUTPUT_NAME="${SCORE_OUTPUT_NAME:-score_llm.json}"
PARALLEL_PER_TASK="${PARALLEL_PER_TASK:-1}"
MATCHING_ORDER="${MATCHING_ORDER:-exact,template,llm}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openai}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5}"
JUDGE_INCLUDE_REASON="${JUDGE_INCLUDE_REASON:-false}"
SCORE_RESUME="${SCORE_RESUME:-true}"
SCORE_SAVE_FREQ="${SCORE_SAVE_FREQ:-20}"

if [[ "${MATCHING_ORDER}" == *"llm"* ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" && "${JUDGE_PROVIDER}" == "openai" ]]; then
    echo "ERROR: OPENAI_API_KEY is required when using llm matcher with openai provider."
    exit 1
  fi
fi

echo "Scoring config:"
echo "  OUT_DIR=${OUT_DIR}"
echo "  SCORE_RESULT_GLOB=${SCORE_RESULT_GLOB}"
echo "  SCORE_OUTPUT_NAME=${SCORE_OUTPUT_NAME}"
echo "  PARALLEL_PER_TASK=${PARALLEL_PER_TASK} (sample workers per result.json)"
echo "  MATCHING_ORDER=${MATCHING_ORDER}"
echo "  JUDGE_PROVIDER=${JUDGE_PROVIDER}"
echo "  JUDGE_MODEL=${JUDGE_MODEL}"
echo "  JUDGE_INCLUDE_REASON=${JUDGE_INCLUDE_REASON}"
echo "  SCORE_RESUME=${SCORE_RESUME}"
echo "  SCORE_SAVE_FREQ=${SCORE_SAVE_FREQ}"
echo

CMD=(
  python mmeval/score.py
  --out_dir "${OUT_DIR}"
  --score_result_glob "${SCORE_RESULT_GLOB}"
  --parallel_per_task "${PARALLEL_PER_TASK}"
  --matching_order "${MATCHING_ORDER}"
  --score_output_name "${SCORE_OUTPUT_NAME}"
  --judge_provider "${JUDGE_PROVIDER}"
  --judge_model "${JUDGE_MODEL}"
  --score_save_freq "${SCORE_SAVE_FREQ}"
)

if [[ "${JUDGE_INCLUDE_REASON}" == "true" ]]; then
  CMD+=(--judge_include_reason)
fi

if [[ "${SCORE_RESUME}" == "false" ]]; then
  CMD+=(--no_score_resume)
fi

"${CMD[@]}"
