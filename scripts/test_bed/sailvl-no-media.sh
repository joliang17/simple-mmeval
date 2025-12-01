#!/bin/bash

export PYTHONPATH=./:$PYTHONPATH

# SAIL-VL-2B - No Media (Pure Text) Test
python mmeval/run.py \
    --dataset local@json \
    --infile test_bed/modality_test/task/no_media.json \
    --out_dir work_dirs/SAIL-VL-2B-no-media \
    --model_name_or_path BytedanceDouyinContent/SAIL-VL-2B \
    --parallel_per_task 1 \
    --gpu_per_parallel 1 \
    --max_new_tokens 1024
