export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --out_dir test_bed/test_blip2_flan_t5 \
    --img_dir test_bed \
    --model_name_or_path Salesforce/blip2-flan-t5-xl \
    --gpu_per_parallel 1 \
    --parallel_per_task 4