export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/Ovis1.6-Llama3.2-3B-multi-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path AIDC-AI/Ovis1.6-Llama3.2-3B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/Ovis1.6-Gemma2-9B-multi-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path AIDC-AI/Ovis1.6-Gemma2-9B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_start.json \
    --dataset local@json \
    --out_dir work_dirs/Ovis1.6-Gemma2-27B-multi-image-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path AIDC-AI/Ovis1.6-Gemma2-27B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 