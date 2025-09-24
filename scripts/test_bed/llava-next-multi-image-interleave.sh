export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/llava-v1.6-mistral-7b-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path llava-hf/llava-v1.6-mistral-7b-hf \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/llava-v1.6-vicuna-7b-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path llava-hf/llava-v1.6-vicuna-7b-hf \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/llava-v1.6-vicuna-13b-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path llava-hf/llava-v1.6-vicuna-13b-hf \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/llama3-llava-next-8b-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path llava-hf/llama3-llava-next-8b-hf \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/llava-v1.6-34b-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path llava-hf/llava-v1.6-34b-hf \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/llava-next-72b-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path llava-hf/llava-next-72b-hf \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/llava-next-110b-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path llava-hf/llava-next-110b-hf \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 512 