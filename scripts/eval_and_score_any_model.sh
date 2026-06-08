#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/eval_and_score_any_model.sh MODEL_ALIAS_OR_PATH ["DATASET [DATASET ...]"] [OUT_ROOT] [MATCHING_ORDER]

MODEL_ALIAS_OR_PATH can be:
  qwen25vl_3b | qwen25vl3b       -> Qwen/Qwen2.5-VL-3B-Instruct
  qwen3vl_4b  | qwen3vl4b        -> Qwen/Qwen3-VL-4B-Instruct
  /path/to/model or HF model id      -> any model supported by mmeval/run.py

Examples:
  scripts/eval_and_score_any_model.sh qwen3vl_4b
  scripts/eval_and_score_any_model.sh qwen25vl_3b "MMStar BLINK"
  scripts/eval_and_score_any_model.sh /path/to/checkpoint-200-merged "MMBench_dev_en MathVista_MINI" work_dirs/ckpt200_llm
  scripts/eval_and_score_any_model.sh qwen3vl_4b "MMStar BLINK" work_dirs/qwen3vl4b_eval exact,template,llm

Environment overrides:
  PROJECT_ROOT=/mnt/bn/algo-masp-nas-arnold2/yijunliang/project
  DATASETS="MMStar BLINK MMBench_dev_en MathVista_MINI"
  OUT_ROOT=work_dirs/my_run
  RUN_SETUP=true|false
  RUN_GPU_SM=true|false
  CACHE_DIR=/mnt/bn/.../cache
  PARALLEL_PER_TASK=auto GPU count / GPU_PER_PARALLEL
  SCORE_PARALLEL_PER_TASK=8
  GPU_PER_PARALLEL=1
  ENABLE_THINKING=true|false
  MAX_NEW_TOKENS=1024
  SAMPLE_NUM=20
  SCORE_OUTPUT_NAME=score_llm_only.json
  MATCHING_ORDER=llm
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
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${ROOT_DIR}/../.." && pwd)}"
cd "${ROOT_DIR}"

MODEL_INPUT="${1:-${MODEL_NAME:-${MODEL_PATH:-}}}"
DATASET_NAME="${2:-${DATASET_NAME:-${DATASETS:-MMStar BLINK MMBench_dev_en MathVista_MINI}}}"
OUT_ROOT="${3:-${OUT_ROOT:-}}"
MATCHING_ORDER="${4:-${MATCHING_ORDER:-llm}}"
DRY_RUN="${DRY_RUN:-false}"

if [[ -z "${MODEL_INPUT}" || -z "${DATASET_NAME}" ]]; then
  usage
  exit 2
fi

resolve_model_name() {
  local key="$1"
  case "${key}" in
    qwen25vl_3b|qwen25vl3b|qwen2.5vl_3b|qwen2.5vl3b)
      printf '%s\n' "Qwen/Qwen2.5-VL-3B-Instruct"
      ;;
    qwen3vl_4b|qwen3vl4b)
      printf '%s\n' "Qwen/Qwen3-VL-4B-Instruct"
      ;;
    *)
      printf '%s\n' "${key}"
      ;;
  esac
}

make_run_key() {
  local model_name="$1"
  local model_path_no_slash="${model_name%/}"
  local model_key
  local parent_key
  model_key="$(basename "${model_path_no_slash}")"
  parent_key="$(basename "$(dirname "${model_path_no_slash}")")"
  if [[ "${model_key}" == checkpoint-* || "${model_key}" == *-merged ]]; then
    model_key="${parent_key}_${model_key}"
  fi
  model_key="$(printf '%s' "${model_key}" | tr -cs '[:alnum:]_.@=-' '_' | sed -e 's/^_//' -e 's/_$//')"
  printf '%s\n' "${model_key}"
}

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "true" ]]; then
    "$@"
  fi
}

MODEL_NAME="$(resolve_model_name "${MODEL_INPUT}")"

if [[ -z "${OUT_ROOT}" ]]; then
  run_key="$(make_run_key "${MODEL_NAME}")"
  OUT_ROOT="${ROOT_DIR}/work_dirs/${run_key}_llm_eval_$(date -u +%Y%m%d_%H%M%S)"
elif [[ "${OUT_ROOT}" != /* ]]; then
  OUT_ROOT="${ROOT_DIR}/${OUT_ROOT}"
fi

SCORE_OUTPUT_NAME="${SCORE_OUTPUT_NAME:-score_llm_only.json}"
SCORE_PARALLEL_PER_TASK="${SCORE_PARALLEL_PER_TASK:-${JUDGE_PARALLEL_PER_TASK:-8}}"

echo "Simple-MMEval infer + judge config:"
echo "  ROOT_DIR=${ROOT_DIR}"
echo "  PROJECT_ROOT=${PROJECT_ROOT}"
echo "  MODEL_INPUT=${MODEL_INPUT}"
echo "  MODEL_NAME=${MODEL_NAME}"
echo "  DATASETS=${DATASET_NAME}"
echo "  OUT_ROOT=${OUT_ROOT}"
echo "  SCORE_OUTPUT_NAME=${SCORE_OUTPUT_NAME}"
echo "  MATCHING_ORDER=${MATCHING_ORDER}"
echo "  SCORE_PARALLEL_PER_TASK=${SCORE_PARALLEL_PER_TASK}"
echo "  JUDGE_PROVIDER=${JUDGE_PROVIDER:-azure_openai}"
echo "  JUDGE_MODEL=${JUDGE_MODEL:-gpt-5.4-mini-2026-03-17}"
echo "  DRY_RUN=${DRY_RUN}"
echo

run_cmd env \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  OUT_ROOT="${OUT_ROOT}" \
  scripts/eval_model.sh "${MODEL_NAME}" "${DATASET_NAME}" "${OUT_ROOT}"

run_cmd env \
  RUN_DIR="${OUT_ROOT}" \
  DATASETS="${DATASET_NAME}" \
  SCORE_OUTPUT_NAME="${SCORE_OUTPUT_NAME}" \
  MATCHING_ORDER="${MATCHING_ORDER}" \
  PARALLEL_PER_TASK="${SCORE_PARALLEL_PER_TASK}" \
  scripts/score_any_model_judge.sh "${OUT_ROOT}" "${DATASET_NAME}" "${SCORE_OUTPUT_NAME}" "${MATCHING_ORDER}"

echo
echo "Done. Outputs:"
echo "  result.json and per-dataset inference logs: ${OUT_ROOT}/<DATASET>/"
echo "  score files: ${OUT_ROOT}/<DATASET>/${SCORE_OUTPUT_NAME}"
