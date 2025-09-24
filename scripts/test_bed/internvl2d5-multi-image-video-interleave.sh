export PYTHONPATH=./:$PYTHONPATH

# InternVL2_5 Base Models
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2_5-1B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2_5-1B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2_5-2B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2_5-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2_5-4B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2_5-4B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2_5-8B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2_5-8B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2_5-26B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2_5-26B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2_5-38B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2_5-38B \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2_5-78B-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2_5-78B \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 512

# InternVL2_5 MPO Models
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2_5-1B-mpo-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2_5-1B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2_5-2B-mpo-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2_5-2B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2_5-4B-mpo-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2_5-4B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2_5-8B-mpo-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2_5-8B-MPO \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2_5-26B-mpo-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2_5-26B-MPO \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2_5-38B-mpo-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2_5-38B-MPO \
    --gpu_per_parallel 2 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/InternVL2_5-78B-mpo-multi-image-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path OpenGVLab/InternVL2_5-78B-MPO \
    --gpu_per_parallel 4 \
    --parallel_per_task 1 \
    --max_new_tokens 512

