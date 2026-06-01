#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/eval_any_model.sh MODEL_PATH [OUT_ROOT] [DATASET...]

Examples:
  scripts/eval_any_model.sh Qwen/Qwen2.5-VL-3B-Instruct
  scripts/eval_any_model.sh /path/to/checkpoint-200-merged work_dirs/ckpt200 MMStar MathVista_MINI

Environment overrides:
  CACHE_DIR=/mnt/bn/.../cache
  DATASETS="MMStar BLINK MMBench_dev_en MathVista_MINI"
  PARALLEL_PER_TASK=4
  GPU_PER_PARALLEL=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

MODEL_PATH="${1:-${MODEL_PATH:-}}"
if [[ -z "${MODEL_PATH}" ]]; then
  usage
  exit 2
fi
shift || true

OUT_ROOT="${1:-${OUT_ROOT:-}}"
if [[ $# -gt 0 ]]; then
  shift || true
fi

if [[ $# -gt 0 ]]; then
  DATASETS="$*"
else
  DATASETS="${DATASETS:-MMStar BLINK MMBench_dev_en MathVista_MINI}"
fi

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

CACHE_DIR="${CACHE_DIR:-/mnt/bn/algo-masp-nas-arnold2/yijunliang/project/cache}"
export HF_HOME="${HF_HOME:-${CACHE_DIR}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export DATASET_DIR="${DATASET_DIR:-${HF_HOME}/simple-mmeval/datasets}"

PARALLEL_PER_TASK="${PARALLEL_PER_TASK:-4}"
GPU_PER_PARALLEL="${GPU_PER_PARALLEL:-1}"

if [[ -z "${OUT_ROOT}" ]]; then
  model_path_no_slash="${MODEL_PATH%/}"
  model_key="$(basename "${model_path_no_slash}")"
  parent_key="$(basename "$(dirname "${model_path_no_slash}")")"
  if [[ "${model_key}" == checkpoint-* || "${model_key}" == *-merged ]]; then
    model_key="${parent_key}_${model_key}"
  fi
  model_key="$(printf '%s' "${model_key}" | tr -cs '[:alnum:]_.@=-' '_' | sed -e 's/^_//' -e 's/_$//')"
  OUT_ROOT="work_dirs/${model_key}"
fi

mkdir -p "${OUT_ROOT}" "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}" "${DATASET_DIR}"

read -r -a dataset_list <<< "${DATASETS}"

echo "Model: ${MODEL_PATH}"
echo "Output: ${OUT_ROOT}"
echo "Datasets: ${dataset_list[*]}"
echo "Cache: ${HF_HOME}"

for dataset in "${dataset_list[@]}"; do
  echo "==> Evaluating ${dataset}"
  python mmeval/run.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset "evalkit@${dataset}" \
    --out_dir "${OUT_ROOT}/${dataset}" \
    --gpu_per_parallel "${GPU_PER_PARALLEL}" \
    --parallel_per_task "${PARALLEL_PER_TASK}" \
    --no_conda
done
