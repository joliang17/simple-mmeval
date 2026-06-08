#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/score_any_model_judge.sh MODEL_OR_RUN_DIR ["DATASET [DATASET ...]"] [SCORE_OUTPUT_NAME] [MATCHING_ORDER]

MODEL_OR_RUN_DIR can be:
  qwen25vl_3b | qwen25vl3b       -> work_dirs/qwen25vl3b_baseline/base
  qwen3vl_4b  | qwen3vl4b        -> work_dirs/qwen3vl4b_mmstar_blink_mmb_mvista_20260606_010007
  /path/to/run_dir or work_dirs/... -> any directory with DATASET/result.json files

Examples:
  scripts/score_any_model_judge.sh qwen25vl_3b
  scripts/score_any_model_judge.sh qwen3vl_4b "MMStar BLINK MMBench_dev_en MathVista_MINI"
  scripts/score_any_model_judge.sh work_dirs/my_model_run "BLINK MMStar" score_llm_only.json exact,template,llm

Environment overrides:
  KEY_CONF=../config/key.conf
  RUN_DIR=/path/to/run_dir
  DATASETS="BLINK MMStar"
  SCORE_OUTPUT_NAME=score_llm_only.json
  MATCHING_ORDER=llm
  PARALLEL_PER_TASK=8
  JUDGE_PROVIDER=azure_openai
  JUDGE_MODEL=gpt-5.4-mini-2026-03-17
  JUDGE_MAX_RETRY=5
  JUDGE_MAX_TOKENS=500
  JUDGE_TEMPERATURE=0.0
  JUDGE_INCLUDE_REASON=true|false
  AZURE_GPT_ENDPOINT=https://aidp-i18ntt-sg.tiktok-row.net/api/modelhub/online/v2/crawl
  DRY_RUN=true
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="./:${PYTHONPATH:-}"

MODEL_OR_RUN_DIR="${1:-${MODEL_OR_RUN_DIR:-}}"
DATASETS="${2:-${DATASETS:-}}"
SCORE_OUTPUT_NAME="${3:-${SCORE_OUTPUT_NAME:-score_llm_only.json}}"
MATCHING_ORDER="${4:-${MATCHING_ORDER:-llm}}"

KEY_CONF="${KEY_CONF:-${PROJECT_ROOT}/config/key.conf}"
PARALLEL_PER_TASK="${PARALLEL_PER_TASK:-8}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-azure_openai}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-mini-2026-03-17}"
JUDGE_MAX_RETRY="${JUDGE_MAX_RETRY:-5}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-500}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
JUDGE_INCLUDE_REASON="${JUDGE_INCLUDE_REASON:-false}"
SCORE_RESUME="${SCORE_RESUME:-true}"
SCORE_PROGRESS_BAR="${SCORE_PROGRESS_BAR:-true}"
SCORE_SAVE_FREQ="${SCORE_SAVE_FREQ:-20}"
SCORE_DEBUG="${SCORE_DEBUG:-false}"
SCORE_DUMP_FAILURES="${SCORE_DUMP_FAILURES:-false}"
DRY_RUN="${DRY_RUN:-false}"

if [[ -z "${MODEL_OR_RUN_DIR}" && -z "${RUN_DIR:-}" ]]; then
  usage
  exit 2
fi

if [[ ! -f "${KEY_CONF}" ]]; then
  echo "ERROR: key config does not exist: ${KEY_CONF}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${KEY_CONF}"
set +a

resolve_run_dir() {
  local key="${1:-}"
  if [[ -n "${RUN_DIR:-}" ]]; then
    printf '%s\n' "${RUN_DIR}"
    return
  fi

  case "${key}" in
    qwen25vl_3b|qwen25vl3b|qwen2.5vl_3b|qwen2.5vl3b)
      printf '%s\n' "${ROOT_DIR}/work_dirs/qwen25vl3b_baseline/base"
      ;;
    qwen3vl_4b|qwen3vl4b)
      printf '%s\n' "${ROOT_DIR}/work_dirs/qwen3vl4b_mmstar_blink_mmb_mvista_20260606_010007"
      ;;
    *)
      if [[ -d "${key}" ]]; then
        cd "${key}" && pwd
      elif [[ -d "${ROOT_DIR}/${key}" ]]; then
        cd "${ROOT_DIR}/${key}" && pwd
      else
        echo "ERROR: unknown model alias or run directory: ${key}" >&2
        echo "       Pass RUN_DIR=/path/to/run_dir or use qwen25vl_3b/qwen3vl_4b." >&2
        exit 1
      fi
      ;;
  esac
}

RUN_DIR="$(resolve_run_dir "${MODEL_OR_RUN_DIR}")"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: run directory does not exist: ${RUN_DIR}" >&2
  exit 1
fi

if [[ "${MATCHING_ORDER}" == *"llm"* && "${JUDGE_PROVIDER}" == "openai" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: OPENAI_API_KEY is required in ${KEY_CONF} for openai llm judge." >&2
  exit 1
fi
if [[ "${MATCHING_ORDER}" == *"llm"* && "${JUDGE_PROVIDER}" == "azure_openai" && -z "${AZURE_GPT_API_KEY:-}" ]]; then
  echo "ERROR: AZURE_GPT_API_KEY is required in ${KEY_CONF} for azure_openai llm judge." >&2
  exit 1
fi

if [[ -z "${DATASETS}" ]]; then
  mapfile -t detected_datasets < <(
    find "${RUN_DIR}" -mindepth 2 -maxdepth 2 -type f -name result.json -printf '%h\n' \
      | while read -r dataset_dir; do basename "${dataset_dir}"; done \
      | sort
  )
  if [[ "${#detected_datasets[@]}" -eq 0 ]]; then
    echo "ERROR: no DATASET/result.json files found under ${RUN_DIR}" >&2
    exit 1
  fi
else
  read -r -a detected_datasets <<< "${DATASETS}"
fi

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "true" ]]; then
    "$@"
  fi
}

echo "Simple-MMEval any-model judge config:"
echo "  ROOT_DIR=${ROOT_DIR}"
echo "  KEY_CONF=${KEY_CONF}"
echo "  MODEL_OR_RUN_DIR=${MODEL_OR_RUN_DIR}"
echo "  RUN_DIR=${RUN_DIR}"
echo "  DATASETS=${detected_datasets[*]}"
echo "  SCORE_OUTPUT_NAME=${SCORE_OUTPUT_NAME}"
echo "  PARALLEL_PER_TASK=${PARALLEL_PER_TASK}"
echo "  MATCHING_ORDER=${MATCHING_ORDER}"
echo "  JUDGE_PROVIDER=${JUDGE_PROVIDER}"
echo "  JUDGE_MODEL=${JUDGE_MODEL}"
echo "  JUDGE_MAX_RETRY=${JUDGE_MAX_RETRY}"
echo "  JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS}"
echo "  JUDGE_TEMPERATURE=${JUDGE_TEMPERATURE}"
echo "  JUDGE_INCLUDE_REASON=${JUDGE_INCLUDE_REASON}"
echo "  SCORE_RESUME=${SCORE_RESUME}"
echo "  SCORE_PROGRESS_BAR=${SCORE_PROGRESS_BAR}"
echo "  SCORE_SAVE_FREQ=${SCORE_SAVE_FREQ}"
echo "  SCORE_DEBUG=${SCORE_DEBUG}"
echo "  SCORE_DUMP_FAILURES=${SCORE_DUMP_FAILURES}"
echo "  DRY_RUN=${DRY_RUN}"
echo

for dataset in "${detected_datasets[@]}"; do
  result_file="${RUN_DIR}/${dataset}/result.json"
  if [[ ! -f "${result_file}" ]]; then
    echo "ERROR: missing result file: ${result_file}" >&2
    exit 1
  fi

  cmd=(
    python mmeval/score.py
    --out-dir "${RUN_DIR}/${dataset}"
    --score-result-glob "result.json"
    --score-output-name "${SCORE_OUTPUT_NAME}"
    --parallel-per-task "${PARALLEL_PER_TASK}"
    --matching-order "${MATCHING_ORDER}"
    --judge-provider "${JUDGE_PROVIDER}"
    --judge-model "${JUDGE_MODEL}"
    --judge-max-retry "${JUDGE_MAX_RETRY}"
    --judge-max-tokens "${JUDGE_MAX_TOKENS}"
    --judge-temperature "${JUDGE_TEMPERATURE}"
    --score-save-freq "${SCORE_SAVE_FREQ}"
  )

  if [[ "${JUDGE_INCLUDE_REASON}" == "true" ]]; then
    cmd+=(--judge-include-reason)
  fi
  if [[ "${SCORE_RESUME}" == "false" ]]; then
    cmd+=(--no-score-resume)
  fi
  if [[ "${SCORE_PROGRESS_BAR}" == "false" ]]; then
    cmd+=(--no-score-progress-bar)
  fi
  if [[ "${SCORE_DEBUG}" == "true" ]]; then
    cmd+=(--score-debug)
  fi
  if [[ "${SCORE_DUMP_FAILURES}" == "true" ]]; then
    cmd+=(--score-dump-failures)
  fi

  run_cmd "${cmd[@]}"
done
