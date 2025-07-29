series_mapping = {
    "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"],
    "visualglm": ["visualglm-6b"]
}

series_infer_env_mapping = {
    "qwenvl2d5": {
        "env": "vllm",
        "infer_file": "qwenvl2d5.py",
    },
    "visualglm": {
        "env": "/root/yuexuanliu/simple-mmeval/envs/visualglm/",
        "infer_file": "visualglm.py",
    }
}