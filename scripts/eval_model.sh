#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/eval_model.sh MODEL_NAME_OR_PATH "DATASET [DATASET ...]" [OUT_ROOT]

Examples:
  scripts/eval_model.sh Qwen/Qwen3-VL-4B-Instruct "MMStar BLINK MMBench_dev_en MathVista_MINI"
  scripts/eval_model.sh /path/to/checkpoint-200-merged "MMStar MathVista_MINI" work_dirs/ckpt200

Environment overrides:
  PROJECT_ROOT=/mnt/bn/algo-masp-nas-arnold2/yijunliang/project
  RUN_SETUP=true|false
  RUN_GPU_SM=true|false
  CACHE_DIR=/mnt/bn/.../cache
  OUT_ROOT=work_dirs/my_run
  PARALLEL_PER_TASK=auto GPU count / GPU_PER_PARALLEL
  GPU_PER_PARALLEL=1
  ENABLE_THINKING=true|false
  MAX_NEW_TOKENS=1024
  SAMPLE_NUM=20
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_NAME="${1:-${MODEL_NAME:-${MODEL_PATH:-}}}"
DATASET_NAME="${2:-${DATASET_NAME:-${DATASETS:-}}}"
OUT_ROOT="${3:-${OUT_ROOT:-}}"

if [[ -z "${MODEL_NAME}" || -z "${DATASET_NAME}" ]]; then
  usage
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/bn/algo-masp-nas-arnold2/yijunliang/project}"
REPO_DIR="${REPO_DIR:-${PROJECT_ROOT}/opsd/simple-mmeval}"
TRAIL_SETUP="${TRAIL_SETUP:-${PROJECT_ROOT}/trail_setup/trail_setup.sh}"
GPU_SM_SETUP="${GPU_SM_SETUP:-${PROJECT_ROOT}/trail_setup/run_gpu_sm.sh}"

CACHE_DIR="${CACHE_DIR:-${PROJECT_ROOT}/cache}"
GPU_PER_PARALLEL="${GPU_PER_PARALLEL:-1}"
RUN_SETUP="${RUN_SETUP:-true}"
RUN_GPU_SM="${RUN_GPU_SM:-true}"

ENABLE_THINKING="${ENABLE_THINKING:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-}"
SAMPLE_NUM="${SAMPLE_NUM:-}"

if [[ -z "${OUT_ROOT}" ]]; then
  model_path_no_slash="${MODEL_NAME%/}"
  model_key="$(basename "${model_path_no_slash}")"
  parent_key="$(basename "$(dirname "${model_path_no_slash}")")"
  if [[ "${model_key}" == checkpoint-* || "${model_key}" == *-merged ]]; then
    model_key="${parent_key}_${model_key}"
  fi
  model_key="$(printf '%s' "${model_key}" | tr -cs '[:alnum:]_.@=-' '_' | sed -e 's/^_//' -e 's/_$//')"
  OUT_ROOT="${REPO_DIR}/work_dirs/${model_key}"
fi

detect_gpu_count() {
  local cvd
  cvd="${CUDA_VISIBLE_DEVICES:-}"

  if [[ -n "${cvd//[[:space:]]/}" ]]; then
    local old_ifs="${IFS}"
    local -a visible_gpus
    IFS=',' read -r -a visible_gpus <<< "${cvd}"
    IFS="${old_ifs}"
    local count=0
    local gpu
    for gpu in "${visible_gpus[@]}"; do
      if [[ -n "${gpu//[[:space:]]/}" ]]; then
        count=$((count + 1))
      fi
    done
    echo "${count}"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | awk '
      /MIG/ && /Device/ { mig += 1 }
      /^GPU [0-9]+:/ { gpu += 1 }
      END {
        if (mig > 0) print mig;
        else print gpu + 0;
      }
    '
    return
  fi

  echo 0
}

if [[ -z "${PARALLEL_PER_TASK:-}" ]]; then
  gpu_count="$(detect_gpu_count)"
  if [[ "${gpu_count}" -gt 0 ]]; then
    PARALLEL_PER_TASK=$((gpu_count / GPU_PER_PARALLEL))
    if [[ "${PARALLEL_PER_TASK}" -lt 1 ]]; then
      PARALLEL_PER_TASK=1
    fi
  else
    PARALLEL_PER_TASK=1
  fi
fi

if [[ "${RUN_SETUP}" == "true" ]]; then
  cd "${PROJECT_ROOT}"
  # shellcheck source=/dev/null
  source "${TRAIL_SETUP}"
fi

if [[ "${RUN_GPU_SM}" == "true" ]]; then
  bash "${GPU_SM_SETUP}"
fi

cd "${REPO_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export DATASET_DIR="${DATASET_DIR:-${HF_HOME}/simple-mmeval/datasets}"

mkdir -p "${OUT_ROOT}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${DATASET_DIR}"

read -r -a datasets <<< "${DATASET_NAME}"

extra_args=()
if [[ -n "${ENABLE_THINKING}" ]]; then
  case "${ENABLE_THINKING}" in
    true|True|TRUE|1|yes|Yes|YES)
      extra_args+=(--enable_thinking)
      ;;
    false|False|FALSE|0|no|No|NO)
      extra_args+=(--no_enable_thinking)
      ;;
    *)
      echo "ENABLE_THINKING must be true or false, got: ${ENABLE_THINKING}" >&2
      exit 2
      ;;
  esac
fi

if [[ -n "${MAX_NEW_TOKENS}" ]]; then
  extra_args+=(--max_new_tokens "${MAX_NEW_TOKENS}")
fi

if [[ -n "${SAMPLE_NUM}" ]]; then
  extra_args+=(--sample_num "${SAMPLE_NUM}")
fi

echo "Model: ${MODEL_NAME}"
echo "Datasets: ${datasets[*]}"
echo "Output root: ${OUT_ROOT}"
echo "Parallel shards: ${PARALLEL_PER_TASK}"
echo "GPUs per shard: ${GPU_PER_PARALLEL}"

for dataset in "${datasets[@]}"; do
  echo "==> Evaluating ${dataset}"
  dataset_out="${OUT_ROOT}/${dataset}"
  mkdir -p "${dataset_out}"
  log_file="${dataset_out}/run_$(date +%Y%m%d_%H%M%S).log"
  echo "Log: ${log_file}"

  if ! python mmeval/run.py \
      --model_name_or_path "${MODEL_NAME}" \
      --dataset "evalkit@${dataset}" \
      --out_dir "${dataset_out}" \
      --gpu_per_parallel "${GPU_PER_PARALLEL}" \
      --parallel_per_task "${PARALLEL_PER_TASK}" \
      --no_conda \
      "${extra_args[@]}" 2>&1 | tee "${log_file}"; then
    echo "ERROR: ${dataset} failed. Last 120 log lines from ${log_file}:" >&2
    tail -n 120 "${log_file}" >&2 || true
    exit 1
  fi
done
