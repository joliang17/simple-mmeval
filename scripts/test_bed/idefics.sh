export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --out_dir test_bed/test_idefics9b \
    --img_dir test_bed \
    --model_name_or_path HuggingFaceM4/idefics-9b-instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 4 \
    --max_new_tokens 128 \
    --do_sample false

python mmeval/run.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --out_dir test_bed/test_idefics2 \
    --img_dir test_bed \
    --model_name_or_path HuggingFaceM4/Idefics2-8b \
    --gpu_per_parallel 1 \
    --parallel_per_task 4 \
    --max_new_tokens 128 \
    --do_sample false

python mmeval/run.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --out_dir test_bed/test_idefics3 \
    --img_dir test_bed \
    --model_name_or_path HuggingFaceM4/Idefics3-8B-Llama3 \
    --gpu_per_parallel 1 \
    --parallel_per_task 4 \
    --max_new_tokens 128 \
    --do_sample false 