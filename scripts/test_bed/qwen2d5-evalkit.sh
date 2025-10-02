export PYTHONPATH=./:$PYTHONPATH
export DATASET_DIR=./mydata
# Before running this script, you need to create .env file in the root directory.
# And set DATASET_DIR in .env as tsv file local directory.

# MM-Math
python mmeval/run.py \
    --dataset evalkit@MM-Math \
    --out_dir work_dirs/qwen2d5_MM_Math \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# MicroBench
python mmeval/run.py \
    --dataset evalkit@MicroBench \
    --out_dir work_dirs/qwen2d5_MicroBench \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# MMMB
python mmeval/run.py \
    --dataset evalkit@MMMB \
    --out_dir work_dirs/qwen2d5_MMMB \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# 3DSRBench (URL)
python mmeval/run.py \
    --dataset https://huggingface.co/datasets/mm-eval/VLMEvalKit/resolve/main/3DSRBench.tsv \
    --out_dir work_dirs/qwen2d5_3DSRBench_test \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --gpu_per_parallel 1 \
    --parallel_per_task 1

# # MM-Math (Local Path)
# python mmeval/run.py \
#     --dataset dataset/MM-Math.tsv \
#     --out_dir work_dirs/qwen2d5_3DSRBench_test \
#     --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
#     --gpu_per_parallel 1 \
#     --parallel_per_task 1
