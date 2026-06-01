#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
CKPT100_MODEL="${CKPT100_MODEL:-/mnt/bn/algo-masp-nas-arnold2/yijunliang/project/opsd/unsup-opsd/work_dirs/opsd_vlm/qwen25vl3b_token_diag_bs8_20260531_204622/Qwen2.5-VL-3B-Instruct-token-diagnostics/checkpoint-100-merged}"

MODEL_PATH="${BASE_MODEL}" \
OUT_ROOT="${OUT_ROOT_BASE:-work_dirs/base_qwen25vl3b}" \
bash scripts/eval_qwen25vl3b.sh

MODEL_PATH="${CKPT100_MODEL}" \
OUT_ROOT="${OUT_ROOT_CKPT100:-work_dirs/ckpt100_qwen25vl3b}" \
bash scripts/eval_qwen25vl3b.sh
