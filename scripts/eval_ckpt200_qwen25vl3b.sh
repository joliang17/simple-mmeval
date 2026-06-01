#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

export HF_HOME="${HF_HOME:-/mnt/bn/algo-masp-nas-arnold2/yijunliang/project/cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export DATASET_DIR="${DATASET_DIR:-${HF_HOME}/simple-mmeval/datasets}"

MODEL_PATH="${MODEL_PATH:-/mnt/bn/algo-masp-nas-arnold2/yijunliang/project/opsd/unsup-opsd/work_dirs/opsd_vlm/qwen25vl3b_token_diag_bs8_20260531_204622/Qwen2.5-VL-3B-Instruct-token-diagnostics/checkpoint-200-merged}"
OUT_ROOT="${OUT_ROOT:-work_dirs/ckpt200_qwen25vl3b}"
PARALLEL_PER_TASK="${PARALLEL_PER_TASK:-4}"
GPU_PER_PARALLEL="${GPU_PER_PARALLEL:-1}"

datasets=(
  MMStar
  BLINK
  MMBench_dev_en
  MathVista_MINI
)

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
