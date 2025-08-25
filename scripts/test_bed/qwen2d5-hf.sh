export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --dataset mmeval_hf@mm-eval/MMMU \
    --split validation \
    --out_dir work_dirs/qwen2d5-mmeval_hf_MMMU_val \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --gpu_per_parallel 2 \
    --parallel_per_task 4 \
    --circular False \
    --resize 512