import os

env_dir = os.getenv('ENV_DIR') or ""

series_mapping = {
    "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"],
    "gemma3": ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"],
    "blip2_flan_t5": ["blip2-flan-t5-xl", "blip2-flan-t5-xxl"],
    "vila": ["VILA1.5-3b", "VILA1.5-13b", "VILA1.5-40b", "Llama-3-VILA1.5-8B"],
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
    "blip2_flan_t5": {
        "env": os.path.join(env_dir, "flan-t5"),
        "infer_file": "blip2_flan_t5.py",
    }, 
    "vila": {
        "env": os.path.join(env_dir, "vila"),
        "infer_file": "vila.py",
    },
}