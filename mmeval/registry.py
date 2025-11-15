import os
from dotenv import load_dotenv

load_dotenv()
env_dir = os.getenv('ENV_DIR') or ""

series_mapping = {
    "blip2_flan_t5": ["blip2-flan-t5-xl", "blip2-flan-t5-xxl"],
    "cambrian": ["cambrian-8b", "cambrian-13b", "cambrian-34b", "cambrian-phi3-3b"],
    "fuyu": ["fuyu-8B"],
    "gemma3": ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it"],
    "llama3d2_vision": ["Llama-3.2-11B-Vision-Instruct", "Llama-3.2-90B-Vision-Instruct"],
    "smolvlm": ["SmolVLM-Instruct", "SmolVLM-Instruct-DPO", "SmolVLM-Instruct-Base","SmolVLM-Sythetic"],
    "llava_ov_1d5": ["LLaVA-OneVision-1.5-8B-Instruct"],
    "glm_4v": ["glm-4v-9b"],
    "instructblip": ["instructblip-vicuna-7b", "instructblip-vicuna-13b", "instructblip-flan-t5-xl", "instructblip-flan-t5-xxl"],
    "internlm_xcomposer": ["internlm-xcomposer-7b"],
    "internvl_chat": ["InternVL-Chat-V1-1", "InternVL-Chat-V1-2", "InternVL-Chat-V1-2-Plus"],
    "internvl_chat1d5": ["Mini-InternVL-Chat-2B-V1-5", "Mini-InternVL-Chat-4B-V1-5", "InternVL-Chat-V1-5"],
    "internvl2": ["InternVL2-1B", "InternVL2-2B", "InternVL2-4B", "InternVL2-8B", "InternVL2-26B", "InternVL2-40B", "InternVL2-Llama3-76B"],
    "internvl2d5": ["InternVL2_5-1B", "InternVL2_5-2B", "InternVL2_5-4B", "InternVL2_5-8B", "InternVL2_5-26B", "InternVL2_5-38B", "InternVL2_5-78B",
                    "InternVL2_5-1B-MPO", "InternVL2_5-2B-MPO", "InternVL2_5-4B-MPO", "InternVL2_5-8B-MPO", "InternVL2_5-26B-MPO", "InternVL2_5-38B-MPO", "InternVL2_5-78B-MPO"],
    "internvl3": ["InternVL3-1B", "InternVL3-2B", "InternVL3-8B", "InternVL3-9B", "InternVL3-14B", "InternVL3-38B", "InternVL3-78B",
                  "InternVL3-1B-Instruct", "InternVL3-2B-Instruct", "InternVL3-8B-Instruct", "InternVL3-9B-Instruct", "InternVL3-14B-Instruct", "InternVL3-38B-Instruct", "InternVL3-78B-Instruct",
                  "InternVL3-1B-Pretrained", "InternVL3-2B-Pretrained", "InternVL3-8B-Pretrained", "InternVL3-9B-Pretrained", "InternVL3-14B-Pretrained", "InternVL3-38B-Pretrained", "InternVL3-78B-Pretrained"],
    "internvl3d5": ["InternVL3_5-1B", "InternVL3_5-2B", "InternVL3_5-4B", "InternVL3_5-8B", "InternVL3_5-14B", "InternVL3_5-GPT-OSS-20B-A4B-Preview", "InternVL3_5-30B-A3B", "InternVL3_5-38B", "InternVL3_5-241B-A28B",
                    "InternVL3_5-1B-MPO", "InternVL3_5-2B-MPO", "InternVL3_5-4B-MPO", "InternVL3_5-8B-MPO", "InternVL3_5-14B-MPO", "InternVL3_5-30B-A3B-MPO", "InternVL3_5-38B-MPO", "InternVL3_5-241B-A28B-MPO",
                    "InternVL3_5-1B-Pretrained", "InternVL3_5-2B-Pretrained", "InternVL3_5-4B-Pretrained", "InternVL3_5-8B-Pretrained", "InternVL3_5-14B-Pretrained", "InternVL3_5-30B-A3B-Pretrained", "InternVL3_5-38B-Pretrained", "InternVL3_5-241B-A28B-Pretrained",
                    "InternVL3_5-1B-Instruct", "InternVL3_5-2B-Instruct", "InternVL3_5-4B-Instruct", "InternVL3_5-8B-Instruct", "InternVL3_5-14B-Instruct", "InternVL3_5-30B-A3B-Instruct", "InternVL3_5-38B-Instruct", "InternVL3_5-241B-A28B-Instruct"],
    "janus": ["Janus-1.3B"],
    "janus_flow": ["JanusFlow-1.3B"],
    "janus_pro": ["Janus-Pro-1B", "Janus-Pro-7B"],
    "llava": ["llava-1.5-7b-hf", "llava-1.5-13b-hf"],
    "llava_next": ["llava-v1.6-mistral-7b-hf", "llava-v1.6-vicuna-7b-hf", "llava-v1.6-vicuna-13b-hf", "llava-v1.6-34b-hf", "llama3-llava-next-8b-hf", "llava-next-72b-hf", "llava-next-110b-hf"],
    "llava_next_interleave": ["llava-next-interleave-qwen-0.5b", "llava-next-interleave-qwen-7b", "llava-next-interleave-qwen-7b-dpo"],
    "llava_ov": ["llava-onevision-qwen2-0.5b-si-hf",  "llava-onevision-qwen2-7b-si-hf", "llava-onevision-qwen2-72b-si-hf", 
    "llava-onevision-qwen2-0.5b-ov-hf","llava-onevision-qwen2-7b-ov-hf", "llava-onevision-qwen2-72b-ov-hf",  
    "llava-onevision-qwen2-7b-ov-chat-hf", "llava-onevision-qwen2-72b-ov-chat-hf"],
    "mantis": ["Mantis-8B-clip-llama3", "Mantis-8B-siglip-llama3"],
    "mantis_fuyu": ["Mantis-8B-Fuyu"],
    "mantis_idefics2": ["Mantis-8B-Idefics2"],
    "mantis_llava": ["Mantis-llava-7b", "Mantis-bakllava-7b"],
    "moondream1": ["moondream1"],
    "moondream2": ["moondream2"],
    "bunnyllama3": ["Bunny-Llama-3-8B-V"],
    "xinyuanvl": ["Xinyuan-VL-2B"],
    "ovis1d5": ["Ovis1.5-Llama3-8B", "Ovis1.5-Gemma2-9B"],
    "phi3v": ["Phi-3.5-vision-instruct", "Phi-3-vision-128k-instruct"],
    "phi4mm": ["Phi-4-multimodal-instruct"],
    "vintern": ["Vintern-1B-v2", "Vintern-1B-v3_5", "Vintern-3B-beta"],
    "xgen": ["xgen-mm-phi3-mini-instruct-interleave-r-v1.5"],
    "ovis1d6": ["Ovis1.6-Llama3.2-3B", "Ovis1.6-Gemma2-9B"],
    "ovis1d6_27b": ["Ovis1.6-Gemma2-27B"],
    # "qwenvl2": ["Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct", "Qwen2-VL-72B-Instruct",
    #             "Qwen2-VL-2B-Instruct-AWQ", "Qwen2-VL-7B-Instruct-AWQ", "Qwen2-VL-72B-Instruct-AWQ",
    #             "Qwen2-VL-2B-Instruct-GPTQ-Int4", "Qwen2-VL-7B-Instruct-GPTQ-Int4", "Qwen2-VL-72B-Instruct-GPTQ-Int4"],
    # "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct",
    #               "Qwen2.5-VL-3B-Instruct-AWQ", "Qwen2.5-VL-7B-Instruct-AWQ", "Qwen2.5-VL-32B-Instruct-AWQ", "Qwen2.5-VL-72B-Instruct-AWQ"],
    # "qwenvl2d5_omni": ["Qwen2.5-Omni-3B", "Qwen2.5-Omni-7B", "Qwen2.5-Omni-7B-AWQ", "Qwen2.5-Omni-7B-GPTQ-Int4"],
    "qwenvl2": ["Qwen2-VL-2B-Instruct", "Qwen2-VL-7B-Instruct", "Qwen2-VL-72B-Instruct"],
    "qwenvl2d5": ["Qwen2.5-VL-3B-Instruct", "Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-32B-Instruct", "Qwen2.5-VL-72B-Instruct"],
    "qwenvl2d5_omni": ["Qwen2.5-Omni-3B", "Qwen2.5-Omni-7B"],
    "videollama2": ["VideoLLaMA2-7B"],
    "vlaa_thinking": ["VLAA-Thinker-Qwen2VL-2B", "VLAA-Thinker-Qwen2VL-7B", "VLAA-Thinker-Qwen2VL-7B-Zero", "VLAA-Thinker-Qwen2.5VL-3B", "VLAA-Thinker-Qwen2.5VL-7B"],
    "r1_onevision": ["R1-Onevision-7B"],
    "wemm": ["WeMM", "WeMM-Chat-CN", "WeMM-Chat-2k-CN"],
    "openai_gpt": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5.1", "gpt-5", "gpt-5-mini"],
    "google_gemini": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite"],
    "anthropic_claude": ["claude-sonnet-4-5", "claude-opus-4-1", "claude-haiku-4-5"],
    "xai_grok": ["grok-4-fast-reasoning", "grok-4-fast-non-reasoning", "grok-4-0709", "grok-2-vision-1212"],
    "doubao_ark": ["doubao-seed-1-6-vision-250815"],
    "hunyuan_vision": ["hunyuan-vision", "hunyuan-t1-vision", "hunyuan-turbos-vision", "hunyuan-large-vision"],
    "sailvl": ["SAIL-VL-2B"],
}

series_infer_env_mapping = {
    "blip2_flan_t5": {
        "env": os.path.join(env_dir, "blip2_flan_t5"),
        "infer_file": "blip2_flan_t5.py",
    },
    "cambrian": {
        "env": os.path.join(env_dir, "cambrian"),
        "infer_file": "cambrian.py",
    },
    "fuyu": {
        "env": os.path.join(env_dir, "fuyu"),
        "infer_file": "fuyu.py",
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
    "internlm_xcomposer": {
        "env": os.path.join(env_dir, "internlm"),
        "infer_file": "internlm_xcomposer.py",
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
    "llava": {
        "env": os.path.join(env_dir, "llava"),
        "infer_file": "llava.py",
    },
    "llava_next": {
        "env": os.path.join(env_dir, "llava_next"),
        "infer_file": "llava_next.py",
    },
    "llava_next_interleave": {
        "env": os.path.join(env_dir, "llava_next_interleave"),
        "infer_file": "llava_next_interleave.py",
    },
    "llava_ov": {
        "env": os.path.join(env_dir, "llava_ov"),
        "infer_file": "llava_ov.py",    
    },
    "llava_ov_1d5": {
        "env": os.path.join(env_dir, "llava_ov_1d5"),
        "infer_file": "llava_ov_1d5.py",
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
    "vintern": {
        "env": os.path.join(env_dir, "vintern"),
        "infer_file": "vintern.py",
    },
    "moondream1": {
        "env": os.path.join(env_dir, "moondream1"),
        "infer_file": "moondream1.py",
    }, 
    "llama3d2_vision": {
        "env": os.path.join(env_dir, "llama3d2_vision"),
        "infer_file": "llama3d2_vision.py",
    }, 
    "smolvlm": {
        "env": os.path.join(env_dir, "smolvlm"),
        "infer_file": "smolvlm.py",
    },
    "moondream2": {
        "env": os.path.join(env_dir, "moondream2"),
        "infer_file": "moondream2.py",
    },
    "bunnyllama3": {
        "env": os.path.join(env_dir, "bunnyllama3"),
        "infer_file": "bunnyllama3.py",
    },
    "xinyuanvl": {
        "env": os.path.join(env_dir, "xinyuanvl"),
        "infer_file": "xinyuanvl.py",
    },
    "xgen": {
        "env": os.path.join(env_dir, "xgen"),
        "infer_file": "xgen.py",
    },
    "paligemma": {
        "env": os.path.join(env_dir, "paligemma"),
        "infer_file": "paligemma.py",
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
    },
    "phi3v": {
        "env": os.path.join(env_dir, "phi3v"),
        "infer_file": "phi3v.py",
    },
    "phi4mm": {
        "env": os.path.join(env_dir, "phi4mm"),
        "infer_file": "phi4mm.py",
    },
    "vlaa_thinking": {
        "env": os.path.join(env_dir, "vlaa_thinking"),
        "infer_file": "vlaa_thinking.py",
    },
    "r1_onevision": {
        "env": os.path.join(env_dir, "r1_onevision"),
        "infer_file": "r1_onevision.py",
    },
    "wemm": {
        "env": os.path.join(env_dir, "wemm"),
        "infer_file": "wemm.py",
    },
    "openai_gpt": {
        "env": os.path.join(env_dir, "openai_gpt"),
        "infer_file": "openai_gpt.py",
    },
    "google_gemini": {
        "env": os.path.join(env_dir, "google_gemini"),
        "infer_file": "google_gemini.py",
    },
    "anthropic_claude": {
        "env": os.path.join(env_dir, "anthropic_claude"),
        "infer_file": "anthropic_claude.py",
    },
    "xai_grok": {
        "env": os.path.join(env_dir, "xai_grok"),
        "infer_file": "xai_grok.py",
    },
    "doubao_ark": {
        "env": os.path.join(env_dir, "doubao_ark"),
        "infer_file": "doubao_ark.py",
    },
    "hunyuan_vision": {
        "env": os.path.join(env_dir, "hunyuan_vision"),
        "infer_file": "hunyuan_vision.py",
    },
    "sailvl": {
        "env": os.path.join(env_dir, "sailvl"),
        "infer_file": "sailvl.py",
    },
}
