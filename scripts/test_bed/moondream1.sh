export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/moondream1-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path vikhyatk/moondream1 \
    --gpu_per_parallel 1 \
    --parallel_per_task 4 \
    --max_new_tokens 512