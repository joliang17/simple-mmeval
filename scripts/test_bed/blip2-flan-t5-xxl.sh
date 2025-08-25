export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --out_dir test_bed/test_blip2_flan_t5_xxl \
    --img_dir test_bed \
    --model_name_or_path Salesforce/blip2-flan-t5-xxl \
    --gpu_per_parallel 2 \
    --parallel_per_task 4