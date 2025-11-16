#!/bin/bash

export PYTHONPATH=./:$PYTHONPATH

# SAIL-VL-2B - Single Image Start Modality Test
python mmeval/run.py \
    --dataset local@json \
    --infile test_bed/modality_test/task/single_image_start.json \
    --img_dir test_bed/modality_test/media/448 \
    --out_dir work_dirs/SAIL-VL-2B-single-image-start \
    --model_name_or_path BytedanceDouyinContent/SAIL-VL-2B \
    --parallel_per_task 1 \
    --gpu_per_parallel 1 \
    --max_new_tokens 1024
