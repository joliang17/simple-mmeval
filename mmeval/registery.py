import os

env_dir = os.getenv('ENV_DIR') or ""

series_mapping = {
    # "blip2_flan_t5": ["blip2-flan-t5-xl", "blip2-flan-t5-xxl"],
    "cambrian": ["cambrian-8b", "cambrian-13b", "cambrian-34b", "cambrian-phi3-3b"],
    "emu2-chat": ["Emu2-Chat", "BAAI/Emu2-Chat"],
    "emu3-chat": ["Emu3-Chat", "BAAI/Emu3-Chat"],
    "gemma3": ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"],
    "glm_4v": ["glm-4v-9b"],
    # "instructblip": ["instructblip-vicuna-7b", "instructblip-vicuna-13b", "instructblip-flan-t5-xl", "instructblip-flan-t5-xxl"],
    "internvl_chat": ["InternVL-Chat-V1-1", "InternVL-Chat-V1-2", "InternVL-Chat-V1-2-Plus"],
    "internvl_chat1d5": ["Mini-InternVL-Chat-2B-V1-5", "Mini-InternVL-Chat-4B-V1-5", "InternVL-Chat-V1-5"],
    "internvl2": ["InternVL2-1B", "InternVL2-2B", "InternVL2-4B", "InternVL2-8B", "InternVL2-26B", "InternVL2-40B", "InternVL2-Llama3-76B"],
    "internvl2d5": ["InternVL2_5-1B", "InternVL2_5-2B", "InternVL2_5-4B", "InternVL2_5-8B", "InternVL2_5-26B", "InternVL2_5-38B", "InternVL2_5-78B",
                    "InternVL2_5-1B-MPO", "InternVL2_5-2B-MPO", "InternVL2_5-4B-MPO", "InternVL2_5-8B-MPO", "InternVL2_5-26B-MPO", "InternVL2_5-38B-MPO", "InternVL2_5-78B-MPO"],
    "internvl3": ["InternVL3-1B", "InternVL3-2B", "InternVL3-8B", "InternVL3-9B", "InternVL3-14B", "InternVL3-38B", "InternVL3-78B",
                  "InternVL3-1B-Instruct", "InternVL3-2B-Instruct", "InternVL3-8B-Instruct", "InternVL3-9B-Instruct", "InternVL3-14B-Instruct", "InternVL3-38B-Instruct", "InternVL3-78B-Instruct",
                  "InternVL3-1B-Pretrained", "InternVL3-2B-Pretrained", "InternVL3-8B-Pretrained", "InternVL3-9B-Pretrained", "InternVL3-14B-Pretrained", "InternVL3-38B-Pretrained", "InternVL3-78B-Pretrained"],
    # "internvl3d5": ["InternVL3_5-1B", "InternVL3_5-2B", "InternVL3_5-4B", "InternVL3_5-8B", "InternVL3_5-14B", "InternVL3_5-GPT-OSS-20B-A4B-Preview", "InternVL3_5-30B-A3B", "InternVL3_5-38B", "InternVL3_5-241B-A28B",
    #                 "InternVL3_5-1B-MPO", "InternVL3_5-2B-MPO", "InternVL3_5-4B-MPO", "InternVL3_5-8B-MPO", "InternVL3_5-14B-MPO", "InternVL3_5-30B-A3B-MPO", "InternVL3_5-38B-MPO", "InternVL3_5-241B-A28B-MPO",
    #                 "InternVL3_5-1B-Pretrained", "InternVL3_5-2B-Pretrained", "InternVL3_5-4B-Pretrained", "InternVL3_5-8B-Pretrained", "InternVL3_5-14B-Pretrained", "InternVL3_5-30B-A3B-Pretrained", "InternVL3_5-38B-Pretrained", "InternVL3_5-241B-A28B-Pretrained",
    #                 "InternVL3_5-1B-Instruct", "InternVL3_5-2B-Instruct", "InternVL3_5-4B-Instruct", "InternVL3_5-8B-Instruct", "InternVL3_5-14B-Instruct", "InternVL3_5-30B-A3B-Instruct", "InternVL3_5-38B-Instruct", "InternVL3_5-241B-A28B-Instruct"],
    # "janus": ["Janus-1.3B"],
    # "janus_flow": ["JanusFlow-1.3B"],
    "janus_pro": ["Janus-Pro-1B", "Janus-Pro-7B"],
    # "llava": ["llava-1.5-7b-hf", "llava-1.5-13b-hf"],
    # "llava_next": ["llava-v1.6-mistral-7b-hf", "llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "llava-v1.6-34b-hf", "llama3-llava-next-8b-hf", "llava-next-72b-hf", "llava-next-110b-hf"],
    "llama-3-mixsensev1_1":["Zero-Vision/Llama-3-MixSenseV1_1", "Llama-3-MixSenseV1_1"],
    "llava_ov": ["llava-onevision-qwen2-0.5b-si-hf",  "llava-onevision-qwen2-7b-si-hf", "llava-onevision-qwen2-72b-si-hf", 
    "llava-onevision-qwen2-0.5b-ov-hf","llava-onevision-qwen2-7b-ov-hf", "llava-onevision-qwen2-72b-ov-hf",  
    "llava-onevision-qwen2-7b-ov-chat-hf", "llava-onevision-qwen2-72b-ov-chat-hf"],
    "mantis": ["Mantis-8B-clip-llama3", "Mantis-8B-siglip-llama3"],
    # "mantis_fuyu": ["Mantis-8B-Fuyu"],
    "mantis_idefics2": ["Mantis-8B-Idefics2"],
    # "mantis_llava": ["Mantis-llava-7b", "Mantis-bakllava-7b"],
    "moondream1": ["moondream1"],
    "moondream2": ["moondream2"],
    "ovis1d5": ["Ovis1.5-Llama3-8B", "Ovis1.5-Gemma2-9B"],
    # "ovis1d6": ["Ovis1.6-Llama3.2-3B", "Ovis1.6-Gemma2-9B"],
    # "ovis1d6_27b": ["Ovis1.6-Gemma2-27B"],
    "qwen-vl": ["Qwen/Qwen-VL"],
    "qwen-vl-chat": ["Qwen/Qwen-VL-Chat"],
    # "qwenvl2": ["Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct", "Qwen2-VL-72B-Instruct",
    #             "Qwen2-VL-2B-Instruct-AWQ", "Qwen2-VL-7B-Instruct-AWQ", "Qwen2-VL-72B-Instruct-AWQ",
    #             "Qwen2-VL-2B-Instruct-GPTQ-Int4", "Qwen2-VL-7B-Instruct-GPTQ-Int4", "Qwen2-VL-72B-Instruct-GPTQ-Int4"],
    # "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct",
    #               "Qwen2.5-VL-3B-Instruct-AWQ", "Qwen2.5-VL-7B-Instruct-AWQ", "Qwen2.5-VL-32B-Instruct-AWQ", "Qwen2.5-VL-72B-Instruct-AWQ"],
    # "qwenvl2d5_omni": ["Qwen2.5-Omni-3B", "Qwen2.5-Omni-7B", "Qwen2.5-Omni-7B-AWQ", "Qwen2.5-Omni-7B-GPTQ-Int4"],
    # "videollama2": ["VideoLLaMA2-7B"]
}

series_infer_env_mapping = {
    "blip2_flan_t5": {
        "env": os.path.join(env_dir, "flan-t5"),
        "infer_file": "blip2_flan_t5.py",
    },
    "cambrian": {
        "env": os.path.join(env_dir, "cambrian"),
        "infer_file": "cambrian.py",
    },
    "emu2-chat": {
        "env": os.path.join(env_dir, "emu2-chat"),
        "infer_file": "emu2-chat.py",
    },
    "emu3-chat": {
        "env": os.path.join(env_dir, "emu3-chat"),
        "infer_file": "emu3-chat.py",
    },
    "gemma3": {
        "env": os.path.join(env_dir, "gemma3"),
        "infer_file": "gemma3.py",
    },
    "glm_4v": {
        "env": os.path.join(env_dir, "glm_4v"),
        "infer_file": "glm_4v.py",
    },
    "instructblip": {
        "env": os.path.join(env_dir, "instructblip"),
        "infer_file": "instructblip.py",
    },
    "internvl2": {
        "env": os.path.join(env_dir, "internvl"),
        "infer_file": "internvl2.py",
    },
    "internvl2d5": {
        "env": os.path.join(env_dir, "internvl"),
        "infer_file": "internvl2d5.py",
    },
    "internvl3": {
        "env": os.path.join(env_dir, "internvl"),
        "infer_file": "internvl3.py",
    },
    "internvl3d5": {
        "env": os.path.join(env_dir, "internvl3d5"),
        "infer_file": "internvl3d5.py",
    },
    "internvl_chat": {
        "env": os.path.join(env_dir, "internvl"),
        "infer_file": "internvl_chat.py",
    },
    "internvl_chat1d5": {
        "env": os.path.join(env_dir, "internvl"),
        "infer_file": "internvl_chat1d5.py",
    },
    "janus": {
        "env": os.path.join(env_dir, "janus"),
        "infer_file": "janus.py",
    },
    "janus_flow": {
        "env": os.path.join(env_dir, "janus"),
        "infer_file": "janusflow.py",
    },
    "janus_pro": {
        "env": os.path.join(env_dir, "janus"),
        "infer_file": "janus_pro.py",
    },
    "llama-3-mixsensev1_1": {
        "env": os.path.join(env_dir, "mixsense"),
        "infer_file": "llama-3-mixsensev1_1.py",
    },
    "llava": {
        "env": os.path.join(env_dir, "llava"),
        "infer_file": "llava.py",
    },
    "llava_next": {
        "env": os.path.join(env_dir, "llava_next"),
        "infer_file": "llava_next.py",
    },
    "llava_ov": {
        "env": os.path.join(env_dir, "llava_ov"),
        "infer_file": "llava_ov.py",
    },
    "mantis": {
        "env": os.path.join(env_dir, "mantis"),
        "infer_file": "mantis.py",
    },
    "mantis_fuyu": {
        "env": os.path.join(env_dir, "mantis"),
        "infer_file": "mantis_fuyu.py",
    },
    "mantis_idefics2": {
        "env": os.path.join(env_dir, "mantis"),
        "infer_file": "mantis_idefics2.py",
    },
    "mantis_llava": {
        "env": os.path.join(env_dir, "mantis"),
        "infer_file": "mantis_llava.py",
    },
    "moondream1": {
        "env": os.path.join(env_dir, "moondream1"),
        "infer_file": "moondream1.py",
    }, 
    "moondream2": {
        "env": os.path.join(env_dir, "moondream2"),
        "infer_file": "moondream2.py",
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
    "paligemma": {
        "env": os.path.join(env_dir, "paligemma"),
        "infer_file": "paligemma.py",
    },
    "qwen-vl": {
        "env": os.path.join(env_dir, "qwen-vl"),
        "infer_file": "qwen-vl.py",
    },
    "qwen-vl-chat": {
        "env": os.path.join(env_dir, "qwen-vl-chat"),
        "infer_file": "qwen-vl-chat.py",
    },
    "qwenvl2": {
        "env": os.path.join(env_dir, "qwenvl"),
        "infer_file": "qwenvl2.py",
    },
    "qwenvl2d5": {
        "env": os.path.join(env_dir, "qwenvl"),
        "infer_file": "qwenvl2d5.py",
    },
    "qwenvl2d5_omni": {
        "env": os.path.join(env_dir, "qwenvl2d5_omni"),
        "infer_file": "qwenvl2d5_omni.py",
    },
    "videollama2": {
        "env": os.path.join(env_dir, "videollama2"),
        "infer_file": "videollama2.py",
    }
}
