export PYTHONPATH=./:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python mmeval/run.py \
    --infile test_bed/modality_test/task/multi_image_video_interleave_long.json \
    --img_dir test_bed/modality_test/media/448 \
    --dataset local@json \
    --out_dir work_dirs/InternVL2_5-1B-resume-test \
    --model_name_or_path OpenGVLab/InternVL2_5-1B \
    --gpu_per_parallel 1 \
    --parallel_per_task 4 \
    --max_new_tokens 512