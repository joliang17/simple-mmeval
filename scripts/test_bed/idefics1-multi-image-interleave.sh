export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/idefics-9b-instruct-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path HuggingFaceM4/idefics-9b-instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/idefics-80b-instruct-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path HuggingFaceM4/idefics-80b-instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --max_new_tokens 512 