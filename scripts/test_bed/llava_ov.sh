export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/llava_ov-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
    --gpu_per_parallel 1 \
    --parallel_per_task 4 \
    --max_new_tokens 512
