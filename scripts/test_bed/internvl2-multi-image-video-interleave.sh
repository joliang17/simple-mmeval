export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2-1B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2-1B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2-2B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2-4B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2-4B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2-8B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2-8B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2-26B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2-26B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2-40B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2-40B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2-Llama3-76B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2-Llama3-76B \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 512
