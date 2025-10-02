export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/Ovis1.5-Llama3-8B-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path AIDC-AI/Ovis1.5-Llama3-8B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/Ovis1.5-Gemma2-9B-single-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path AIDC-AI/Ovis1.5-Gemma2-9B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1