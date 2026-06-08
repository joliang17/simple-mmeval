#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/merge_eval_score_models.sh CHECKPOINT_OR_RUN_DIR [CHECKPOINT_OR_RUN_DIR ...]

Merge one or more LoRA checkpoints, run Simple-MMEval inference on multiple
datasets, then score outputs with the configured LLM judge.

Default datasets:
  MMStar BLINK MMBench_dev_en MathVista_MINI

Examples:
  scripts/merge_eval_score_models.sh ../qgen_opsd/work_dirs/run_a
  DATASETS="MMStar BLINK" scripts/merge_eval_score_models.sh /path/to/checkpoint-200 /path/to/checkpoint-400
  MATCHING_ORDER=llm SCORE_OUTPUT_NAME=score_llm_only.json scripts/merge_eval_score_models.sh /path/to/run
  RUN_MERGE=false scripts/merge_eval_score_models.sh /path/to/already-merged-model
  DRY_RUN=true scripts/merge_eval_score_models.sh /path/to/run

Environment overrides:
  DATASETS="MMStar BLINK MMBench_dev_en MathVista_MINI"
  MATCHING_ORDER=llm
  SCORE_OUTPUT_NAME=score_llm_only.json
  OUT_ROOT=/path/to/eval_outputs
  MERGED_ROOT=/path/to/merged_models
  MERGED_PATH=/path/to/single_merged_model
  RUN_MERGE=true|false
  SKIP_MERGE_IF_EXISTS=true|false
  MERGE_SCRIPT=/path/to/qgen_opsd/eval/merge.sh
  BASE_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
  PROJECT_ROOT=/mnt/bn/algo-masp-nas-arnold2/yijunliang/project
  UPDATE_SUMMARY=true|false
  SUMMARY_PATH=eval_summary.csv
  DRY_RUN=true|false

Inference and judge env from eval_and_score_any_model.sh are passed through,
including RUN_SETUP, RUN_GPU_SM, PARALLEL_PER_TASK, GPU_PER_PARALLEL,
ENABLE_THINKING, MAX_NEW_TOKENS, SAMPLE_NUM, SCORE_PARALLEL_PER_TASK,
JUDGE_PROVIDER, JUDGE_MODEL, AZURE_GPT_ENDPOINT, and AZURE_GPT_API_KEY.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
WORKSPACE_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${ROOT_DIR}/../.." && pwd)}"
MERGE_SCRIPT="${MERGE_SCRIPT:-${WORKSPACE_ROOT}/qgen_opsd/eval/merge.sh}"
DATASETS="${DATASETS:-MMStar BLINK MMBench_dev_en MathVista_MINI}"
MATCHING_ORDER="${MATCHING_ORDER:-llm}"
SCORE_OUTPUT_NAME="${SCORE_OUTPUT_NAME:-score_llm_only.json}"
RUN_MERGE="${RUN_MERGE:-true}"
SKIP_MERGE_IF_EXISTS="${SKIP_MERGE_IF_EXISTS:-true}"
UPDATE_SUMMARY="${UPDATE_SUMMARY:-false}"
SUMMARY_PATH="${SUMMARY_PATH:-${ROOT_DIR}/eval_summary.csv}"
DRY_RUN="${DRY_RUN:-false}"

if [[ -n "${MERGED_PATH:-}" && $# -ne 1 ]]; then
  echo "MERGED_PATH can only be used with a single input." >&2
  exit 2
fi

if [[ "${RUN_MERGE}" == "true" && ! -f "${MERGE_SCRIPT}" ]]; then
  echo "Merge script not found: ${MERGE_SCRIPT}" >&2
  exit 2
fi

if [[ "${RUN_MERGE}" == "true" ]]; then
  MERGE_WORKDIR="$(cd "$(dirname "${MERGE_SCRIPT}")"/.. && pwd)"
else
  MERGE_WORKDIR=""
fi

cd "${ROOT_DIR}"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "true" ]]; then
    "$@"
  fi
}

run_merge_cmd() {
  printf '+ cd %q &&' "${MERGE_WORKDIR}"
  printf ' %q' env "BASE_MODEL=${BASE_MODEL:-}" bash "${MERGE_SCRIPT}" "$1" "$2"
  printf '\n'
  if [[ "${DRY_RUN}" != "true" ]]; then
    (cd "${MERGE_WORKDIR}" && env BASE_MODEL="${BASE_MODEL:-}" bash "${MERGE_SCRIPT}" "$1" "$2")
  fi
}

safe_name() {
  local value="$1"
  value="${value%/}"
  if [[ "$(basename "${value}")" == checkpoint-* ]]; then
    value="$(basename "$(dirname "${value}")")_$(basename "${value}")"
  else
    value="$(basename "${value}")"
  fi
  printf '%s' "${value}" | tr -cs '[:alnum:]_.@=-' '_' | sed -e 's/^_//' -e 's/_$//'
}

abs_path_if_local() {
  local path="$1"
  local dir
  if [[ "${path}" == /* ]]; then
    printf '%s\n' "${path}"
    return
  fi

  dir="$(dirname "${path}")"
  if [[ -d "${dir}" ]]; then
    printf '%s/%s\n' "$(cd "${dir}" && pwd)" "$(basename "${path}")"
    return
  fi

  printf '%s\n' "${path}"
}

resolve_checkpoint() {
  local path="$1"
  if [[ -f "${path}/adapter_model.safetensors" || -f "${path}/adapter_model.bin" ]]; then
    printf '%s\n' "${path}"
    return
  fi

  local latest
  latest=$(find "${path}" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -1)
  if [[ -n "${latest}" && (-f "${latest}/adapter_model.safetensors" || -f "${latest}/adapter_model.bin") ]]; then
    printf '%s\n' "${latest}"
    return
  fi

  printf '%s\n' "${path}"
}

merged_path_for() {
  local checkpoint="$1"
  local merged_path
  if [[ -n "${MERGED_PATH:-}" ]]; then
    merged_path="${MERGED_PATH}"
  elif [[ -n "${MERGED_ROOT:-}" ]]; then
    merged_path="${MERGED_ROOT%/}/$(safe_name "${checkpoint}")-merged"
  else
    merged_path="${checkpoint}-merged"
  fi
  abs_path_if_local "${merged_path}"
}

out_root_for() {
  local model_path="$1"
  local key
  key="$(safe_name "${model_path}")"
  if [[ -n "${OUT_ROOT:-}" ]]; then
    if [[ $# -eq 2 && "$2" == "single" ]]; then
      printf '%s\n' "${OUT_ROOT}"
    else
      printf '%s\n' "${OUT_ROOT%/}/${key}"
    fi
  else
    printf '%s\n' "${ROOT_DIR}/work_dirs/${key}_llm_eval_$(date -u +%Y%m%d_%H%M%S)"
  fi
}

echo "Merge + eval + score config:"
echo "  ROOT_DIR=${ROOT_DIR}"
echo "  PROJECT_ROOT=${PROJECT_ROOT}"
echo "  MERGE_SCRIPT=${MERGE_SCRIPT}"
echo "  DATASETS=${DATASETS}"
echo "  MATCHING_ORDER=${MATCHING_ORDER}"
echo "  SCORE_OUTPUT_NAME=${SCORE_OUTPUT_NAME}"
echo "  RUN_MERGE=${RUN_MERGE}"
echo "  SKIP_MERGE_IF_EXISTS=${SKIP_MERGE_IF_EXISTS}"
echo "  DRY_RUN=${DRY_RUN}"
echo

input_count=$#
for input_path in "$@"; do
  checkpoint="$(abs_path_if_local "$(resolve_checkpoint "${input_path}")")"
  if [[ "${RUN_MERGE}" == "true" ]]; then
    model_path="$(merged_path_for "${checkpoint}")"
  else
    model_path="$(abs_path_if_local "${input_path}")"
  fi

  if [[ -n "${OUT_ROOT:-}" && "${input_count}" -eq 1 ]]; then
    run_out_root="$(out_root_for "${model_path}" single)"
  else
    run_out_root="$(out_root_for "${model_path}")"
  fi

  echo "Input: ${input_path}"
  echo "  checkpoint=${checkpoint}"
  echo "  model_path=${model_path}"
  echo "  out_root=${run_out_root}"

  if [[ "${RUN_MERGE}" == "true" ]]; then
    if [[ "${SKIP_MERGE_IF_EXISTS}" == "true" && -f "${model_path}/config.json" ]]; then
      echo "  merge=skip existing merged model"
    else
      run_merge_cmd "${checkpoint}" "${model_path}"
    fi
  fi

  run_cmd env \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    DATASETS="${DATASETS}" \
    SCORE_OUTPUT_NAME="${SCORE_OUTPUT_NAME}" \
    MATCHING_ORDER="${MATCHING_ORDER}" \
    scripts/eval_and_score_any_model.sh "${model_path}" "${DATASETS}" "${run_out_root}" "${MATCHING_ORDER}"

  echo
done

if [[ "${UPDATE_SUMMARY}" == "true" ]]; then
  run_cmd python scripts/update_eval_summary.py --root work_dirs --out "${SUMMARY_PATH}"
fi

echo "Done. Per-dataset outputs are under each run's OUT_ROOT/<DATASET>/."
