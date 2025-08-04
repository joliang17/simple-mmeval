export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --out_dir test_bed/test_minimonkey \
    --img_dir test_bed \
    --model_name_or_path MiniMonkey \
    --max_new_tokens 512 \
    --do_sample False \
    --gpu_per_parallel 1 \
    --parallel_per_task 4 