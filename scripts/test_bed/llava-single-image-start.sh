export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/llava-1.5-7b-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path llava-hf/llava-1.5-7b-hf \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/llava-1.5-13b-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path llava-hf/llava-1.5-13b-hf \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512