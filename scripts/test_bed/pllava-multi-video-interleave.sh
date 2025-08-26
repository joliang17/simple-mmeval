export PYTHONPATH=./:$PYTHONPATH
export ENV_DIR=/root/zhichengjiang/simple-mmeval/envs

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/pllava-7b-multi-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path /root/zhichengjiang/simple-mmeval/models/pllava-7b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/pllava-13b-multi-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path /root/zhichengjiang/simple-mmeval/models/pllava-13b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/pllava-34b-multi-video-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path /root/zhichengjiang/simple-mmeval/models/pllava-34b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512