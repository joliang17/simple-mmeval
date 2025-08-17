import os

env_dir = os.getenv('ENV_DIR') or ""

series_mapping = {
    "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"],
    "gemma3": ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"],
    "h2ovl-mississippi": ["h2ovl-mississippi-800m","h2ovl-mississippi-2b"]
}

series_infer_env_mapping = {
    "qwenvl2d5": {
        "env": os.path.join(env_dir, "vllm"),
        "infer_file": "qwenvl2d5.py",
    }, 
    "gemma3": {
        "env": os.path.join(env_dir, "gemma3"),
        "infer_file": "gemma3.py",
    },
    "h2ovl-mississippi": {
        "env": "/root/yuexuanliu/simple-mmeval/envs/h2ovl-mississippi",
        "infer_file": "h2ovl-mississippi.py",
    }
}