import re
import copy
import torch

from transformers import AutoModel, AutoTokenizer

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.helper import transform_image

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 512, "do_sample": False, "low_cpu_mem_usage": True}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)
    
    def load_model(self, args):
        self.model = AutoModel.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            **self.model_kwargs).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)

    def run_sample(self, sample:dict):

        ori_sample = copy.deepcopy(sample)
        parsed_sample = self.parse_input(sample)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(parsed_sample)
        else:
            raise NotImplementedError("Scoring is not supported")
        
        return ori_sample
    
    def _generate_response(self, data):
        # keep same with sample code: https://github.com/Yuliang-Liu/Monkey/blob/main/project/mini_monkey/demo.py
        generation_config = dict(do_sample=self.gen_kwargs["do_sample"], max_new_tokens=self.gen_kwargs["max_new_tokens"])

        # pixel_values/ target_aspect_ratio/ question information extract in "parse_input" function
        response, history = self.model.chat(self.tokenizer, data["pixel_values"], data["target_aspect_ratio"], data["question"], generation_config, history=None, return_history=True)

        return response
    
    def parse_input(self, sample:dict):
        parsed_sample = copy.deepcopy(sample)

        question = sample["prompt"]
        # extract placeholder
        placeholders = re.findall(r'<(?:image|video)>', question)
        assert len(placeholders) == 1 and placeholders[0] == constants.image, f"Minimonkey supports one image, but got {len(placeholder)}"
        placeholder = placeholders[0]

        # remove the placeholder in the question
        question = question.replace(placeholder, "").strip()
        parsed_sample["question"] = question

        image = sample["media"][0]
        pixel_values, target_aspect_ratio = transform_image(image, min_num=4, max_num=12)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        parsed_sample["pixel_values"] = pixel_values
        parsed_sample["target_aspect_ratio"] = target_aspect_ratio

        return parsed_sample

if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()

        
