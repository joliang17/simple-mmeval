export PYTHONPATH=./:$PYTHONPATH

python mmeval/infer/llava_next_video.py \
    --infile test_bed/image.json \
    --dataset local@json \
    --out_dir test_bed/test_llava_ov \
    --img_dir test_bed \
    --model_name_or_path llava-hf/LLaVA-NeXT-Video-7B-hf