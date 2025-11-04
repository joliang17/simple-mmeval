export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_interleave.json \
    --dataset local@json \
    --out_dir work_dirs/Xinyuan-VL-2B-multi-image-interleave \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path Cylingo/Xinyuan-VL-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1