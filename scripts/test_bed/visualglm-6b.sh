export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --out_dir test_bed/test_visualglm-6b \
    --img_dir test_bed \
    --model_name_or_path visualglm-6b \
    --gpu_per_parallel 1 \
    --parallel_per_task 4