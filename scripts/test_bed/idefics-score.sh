export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/image-qca.json \
    --dataset local@json \
    --out_dir test_bed/test_idefics9b_score \
    --img_dir test_bed \
    --model_name_or_path HuggingFaceM4/idefics-9b-instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --score_target

python mmeval/run.py \
    --infile test_bed/image-qca.json \
    --dataset local@json \
    --out_dir test_bed/test_idefics2_score \
    --img_dir test_bed \
    --model_name_or_path HuggingFaceM4/Idefics2-8b \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --score_target

python mmeval/run.py \
    --infile test_bed/image-qca.json \
    --dataset local@json \
    --out_dir test_bed/test_idefics3_score \
    --img_dir test_bed \
    --model_name_or_path HuggingFaceM4/Idefics3-8B-Llama3 \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --score_target 