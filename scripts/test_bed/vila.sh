#!/bin/bash

export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --model_name_or_path vila \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/VILA-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Efficient-Large-Model/VILA1.5-3b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1
