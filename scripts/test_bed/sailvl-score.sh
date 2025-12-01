#!/bin/bash

# ⚠️ NOTE: Scoring mode is NOT supported for SAIL-VL-2B
# This script will fail with a NotImplementedError.
# SAIL-VL uses a custom chat interface that requires special image token replacement,
# making it incompatible with the standard scoring mechanism.
#
# For testing SAIL-VL-2B, use generation mode scripts instead:
# - sailvl-single-image-start.sh
# - sailvl-multi-image-start.sh
# - sailvl-multi-image-interleave.sh
# - sailvl-no-media.sh

export PYTHONPATH=./:$PYTHONPATH

# SAIL-VL-2B - Multiple Choice Scoring Test (NOT SUPPORTED)
python mmeval/run.py \
    --dataset local@json \
    --infile test_bed/image-qca.json \
    --img_dir test_bed \
    --out_dir work_dirs/SAIL-VL-2B-score \
    --model_name_or_path BytedanceDouyinContent/SAIL-VL-2B \
    --parallel_per_task 1 \
    --gpu_per_parallel 1 \
    --score_target
