import os

env_dir = os.getenv('ENV_DIR') or ""

series_mapping = {
    "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"],
    "gemma3": ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"],
    "llava": ["llava-1.5-7b-hf", "llava-1.5-13b-hf"],
    "llava_next": ["llava-v1.6-mistral-7b-hf", "llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "llava-v1.6-34b-hf", "llama3-llava-next-8b-hf", "llava-next-72b-hf", "llava-next-110b-hf"],
    "glm_4v": ["glm-4v-9b"],
    "ovis1d5": ["Ovis1.5-Llama3-8B", "Ovis1.5-Gemma2-9B"],
    "ovis1d6": ["Ovis1.6-Llama3.2-3B", "Ovis1.6-Gemma2-9B"],
    "ovis1d6_27b": ["Ovis1.6-Gemma2-27B"],
    "blip2_flan_t5": ["blip2-flan-t5-xl", "blip2-flan-t5-xxl"],
    "moondream1": ["moondream1"],
    "moondream2": ["moondream2"],
    "minimonkey": ["MiniMonkey"]
}

series_infer_env_mapping = {
    "qwenvl2d5": {
        "env": os.path.join(env_dir, "qwenvl2d5"),
        "infer_file": "qwenvl2d5.py",
    },
    "gemma3": {
        "env": os.path.join(env_dir, "gemma3"),
        "infer_file": "gemma3.py",
    },
    "llava": {
        "env": os.path.join(env_dir, "llava"),
        "infer_file": "llava.py",
    },
    "llava_next": {
        "env": os.path.join(env_dir, "llava_next"),
        "infer_file": "llava_next.py",
    },
    "glm_4v": {
        "env": os.path.join(env_dir, "glm_4v"),
        "infer_file": "glm_4v.py",
    },
    "ovis1d5": {
        "env": os.path.join(env_dir, "ovis1d5"),
        "infer_file": "ovis1d5.py",
    },
    "ovis1d6": {
        "env": os.path.join(env_dir, "ovis1d6"),
        "infer_file": "ovis1d6.py",
    },
    "ovis1d6_27b": {
        "env": os.path.join(env_dir, "ovis1d6_27b"),
        "infer_file": "ovis1d6.py",
    },
    "blip2_flan_t5": {
        "env": os.path.join(env_dir, "flan-t5"),
        "infer_file": "blip2_flan_t5.py",
    },
    "moondream1": {
        "env": os.path.join(env_dir, "moondream1"),
        "infer_file": "moondream1.py",
    }, 
    "moondream2": {
        "env": os.path.join(env_dir, "moondream2"),
        "infer_file": "moondream2.py",
    },
    "minimonkey": {
        "env": os.path.join(env_dir, "minimonkey"),
        "infer_file": "minimonkey.py",
    }
}