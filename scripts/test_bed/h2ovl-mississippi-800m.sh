export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --out_dir test_bed/test_h2ovl-mississippi-800m \
    --img_dir test_bed \
    --model_name_or_path h2ovl-mississippi-800m \
    --gpu_per_parallel 1 \
    --parallel_per_task 4