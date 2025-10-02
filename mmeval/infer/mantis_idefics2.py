import os, sys, importlib
d = os.path.abspath(os.path.dirname(__file__))
popped = sys.path.pop(0) if sys.path and os.path.abspath(sys.path[0]) == d else None
try:
    m = importlib.import_module('mantis')  
    sys.modules['mantis'] = m        
finally:
    if popped is not None:
        sys.path.insert(0, popped)

import re
import copy
import torch

from transformers import AutoProcessor, AutoModelForVision2Seq

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 1024, "do_sample": False, "num_beams": 1}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path) # do_image_splitting is False by default
        self.model = AutoModelForVision2Seq.from_pretrained(args.model_name_or_path, **self.model_kwargs)

    def _parse_input(self, sample:dict):
        prompt = sample["prompt"]
        # placeholder <>, can be image, video, audio, etc.
        q_chunks = re.split(r'(<(?:image|video)>)', prompt)

        messages = [
            {
                "role": "user",
                "content": []
            }
        ]

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                messages[0]["content"].append(
                    {"type": "image"}
                )       
            else:
                messages[0]["content"].append(
                    {
                        "type": "text",
                        "text": chunk
                    }
                )
        print(messages)
        return messages

    def _generate_response(self, inputs):
        # Generate
        generated_ids = self.model.generate(**inputs, **self.gen_kwargs)
        response = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return response[0]
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        
        messages = self._parse_input(ori_sample)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        images = sample["media"]
        inputs = self.processor(text=prompt, images=images, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(inputs)
        else:
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
