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

from mantis.models.mllava import chat_mllava
from mantis.models.mllava import MLlavaProcessor, LlavaForConditionalGeneration

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"attn_implementation": "flash_attention_2", "device_map": "auto"}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args)

        super().__init__(args)

    def load_model(self, args):
        # load processor and model
        self.processor = MLlavaProcessor.from_pretrained(args.model_name_or_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(args.model_name_or_path, torch_dtype=self.dtype, **self.model_kwargs)

    def _generate_response(self, text, images):
        response, _ = chat_mllava(text, images, self.model, self.processor, **self.gen_kwargs)

        return response
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        text = sample["prompt"]
        images = sample["media"]

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(text, images)
        else:
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
