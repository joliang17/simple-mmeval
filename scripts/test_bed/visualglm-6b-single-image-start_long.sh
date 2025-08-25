export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_image_start_long.json \
    --dataset local@json \
    --out_dir work_dirs/visualglm-6b-single-image-start_long \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path visualglm-6b \
    --gpu_per_parallel 1 \
    --parallel_per_task 4