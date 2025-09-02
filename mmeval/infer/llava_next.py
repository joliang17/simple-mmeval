import re
import copy
import torch

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"attn_implementation": "flash_attention_2", "device_map": "auto", "low_cpu_mem_usage": True}
        self.default_gen_kwargs = {"max_new_tokens": 100}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        self.processor = LlavaNextProcessor.from_pretrained(args.model_name_or_path)

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_name_or_path, 
            torch_dtype=self.dtype, 
            load_in_4bit=False,
            **self.model_kwargs
        ) 
        
    def _parse_input(self, sample:dict):
        prompt = sample["prompt"]
        # placeholder <>, can be image, video, etc.
        q_chunks = re.split(r'(<(?:image|video)>)', prompt)

        conversation = [
            {
                "role": "user",
                "content": []
            }
        ]

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                conversation[0]["content"].append(
                    {
                        "type": "image"
                    }
                )       
            else:
                conversation[0]["content"].append(
                    {
                        "type": "text",
                        "text": chunk
                    }
                )

        return conversation

    def _generate_response(self, inputs):
        output = self.model.generate(**inputs, **self.gen_kwargs)

        return self.processor.decode(output[0], skip_special_tokens=True)
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        conversation = self._parse_input(ori_sample)
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        images = ori_sample["media"]
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(self.model.device)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(inputs)
        else:
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
