export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/llava_next_video-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path llava-hf/LLaVA-NeXT-Video-7B-hf \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512