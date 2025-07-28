series_mapping = {
    "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"],
    "h2ovl-mississippi": ["h2ovl-mississippi-800m","h2ovl-mississippi-2b"]
}

series_infer_env_mapping = {
    "qwenvl2d5": {
        "env": "vllm",
        "infer_file": "qwenvl2d5.py",
    },
    "h2ovl-mississippi": {
        "env": "/root/yuexuanliu/simple-mmeval/envs/h2ovl-mississippi",
        "infer_file": "h2ovl-mississippi.py",
    }
}