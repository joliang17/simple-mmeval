import re
import copy
import torch
import json
import os
from PIL import Image

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from mmeval.infer.task import Task
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from transformers.models.qwen2 import Qwen2Config
        

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"low_cpu_mem_usage": True}
        self.default_gen_kwargs = {"max_new_tokens": 2500, "do_sample": True, "top_k": 1}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        
        super().__init__(args)

    """
    The Parrot-7B model uses a custom model type "parrot_qwen2" in its configuration that is not 
    recognized by the transformers library. The recognized model type for Parrot is "qwen2".

    Thus, we need to override the model_type in the config (Change "parrot_qwen2" to "qwen2") 
    before loading the model.
    """
    def load_model(self, args):
        # Load config manually and modify model_type from parrot_qwen2 to qwen2
        if os.path.isdir(args.model_name_or_path):
            config_path = os.path.join(args.model_name_or_path, "config.json")
        else:
            # For HuggingFace models, we need to download the config file
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(repo_id=args.model_name_or_path, filename="config.json")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        config_dict["model_type"] = "qwen2"
        if "architectures" in config_dict:
            config_dict["architectures"] = ["Qwen2ForCausalLM"]
        
        config = Qwen2Config.from_dict(config_dict)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            **self.model_kwargs
        ).to(self.device).eval()
        
    def _parse_input(self, sample: dict):
        prompt = sample["prompt"]
        query = prompt.replace("<image>", "")
        return query

    def _generate_response(self, inputs):
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        query = self._parse_input(ori_sample)
        
        image_path = ori_sample['media'][0]
        
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "image": image, "content": query}],
            add_generation_prompt=True, 
            tokenize=True, 
            return_tensors="pt",
            return_dict=True
        )

        inputs = inputs.to(self.device)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(inputs)
        else:
            # Handle scoring mode if needed
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
