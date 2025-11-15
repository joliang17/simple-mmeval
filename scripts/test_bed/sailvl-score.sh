export PYTHONPATH=./:$PYTHONPATH

# SAIL-VL-2B - Multiple Choice Scoring Test
python mmeval/run.py \
    --infile test_bed/image-qca.json \
    --dataset local@json \
    --out_dir work_dirs/SAIL-VL-2B-score \
    --img_dir test_bed \
    --model_name_or_path BytedanceDouyinContent/SAIL-VL-2B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1 \
    --score_target
