export PYTHONPATH=./:$PYTHONPATH

# SAIL-VL-2B - Single Image Generation Test
python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/SAIL-VL-2B-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path BytedanceDouyinContent/SAIL-VL-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 1024
