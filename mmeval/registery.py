series_mapping = {
    "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"],
    "llava_next_video": ["LLaVA-NeXT-Video-32B-Qwen",
                         "LLaVA-NeXT-Video-32B-Qwen_multi_frame",
                         "LLaVA-NeXT-Video-7B",
                         "LLaVA-NeXT-Video-7B_multi_frame",
                         "LLaVA-NeXT-Video-7B-DPO",
                         "LLaVA-NeXT-Video-7B-DPO_multi_frame"]
}

series_infer_env_mapping = {
    "qwenvl2d5": {
        "env": "vllm",
        "infer_file": "qwenvl2d5.py",
    },
    "llava_next_video": {
        "env": os.path.join(env_dir, "llava_next_video"),
        "infer_file": "llava_next_video.py",
    }
}