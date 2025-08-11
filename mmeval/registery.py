import os

env_dir = os.getenv('ENV_DIR') or ""

series_mapping = {
    "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"],
    "gemma3": ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"],
    "idefics": [
        "idefics-80b-instruct",
        "idefics-9b-instruct",
        "Idefics2-8b",
        "Idefics3-8B-Llama3",
    ]
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
    "idefics": {
        "env": os.path.join(env_dir, "idefics"),
        "infer_file": "idefics.py",
    }
}