#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

export HF_HOME="${HF_HOME:-/mnt/bn/algo-masp-nas-arnold2/yijunliang/project/cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export DATASET_DIR="${DATASET_DIR:-${HF_HOME}/simple-mmeval/datasets}"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a Hugging Face model id or local model path}"
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT to the output directory for this run}"
PARALLEL_PER_TASK="${PARALLEL_PER_TASK:-4}"
GPU_PER_PARALLEL="${GPU_PER_PARALLEL:-1}"

# DATASETS env var (space-separated) overrides the default list
if [[ -n "${DATASETS:-}" ]]; then
    read -r -a datasets <<< "${DATASETS}"
else
    datasets=(
      MMStar
      BLINK
      MMBench_dev_en
      MathVista_MINI
    )
fi

mkdir -p "${OUT_ROOT}"

for dataset in "${datasets[@]}"; do
  python mmeval/run.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset "evalkit@${dataset}" \
    --out_dir "${OUT_ROOT}/${dataset}" \
    --gpu_per_parallel "${GPU_PER_PARALLEL}" \
    --parallel_per_task "${PARALLEL_PER_TASK}" \
    --no_conda
done
