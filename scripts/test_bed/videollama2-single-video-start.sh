export PYTHONPATH=./:$PYTHONPATH

python mmeval/run.py \
    --infile test_bed/modality_test/task/single_video_start.json \
    --dataset local@json \
    --out_dir work_dirs/VideoLLaMA2-7B-single-video-start \
    --img_dir test_bed/modality_test/media/448 \
    --model_name_or_path DAMO-NLP-SG/VideoLLaMA2-7B \
    --gpu_per_parallel 1 \
    --parallel_per_task 1