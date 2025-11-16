#!/bin/bash

export PYTHONPATH=./:$PYTHONPATH

# SAIL-VL-2B - Multiple Choice Scoring Test
python mmeval/run.py \
    --dataset local@json \
    --infile test_bed/image-qca.json \
    --img_dir test_bed \
    --out_dir work_dirs/SAIL-VL-2B-score \
    --model_name_or_path BytedanceDouyinContent/SAIL-VL-2B \
    --parallel_per_task 1 \
    --gpu_per_parallel 1 \
    --score_target
