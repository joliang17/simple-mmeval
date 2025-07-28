import re
import copy
import torch
import numpy as np
from transformers import AutoConfig, AutoModel, AutoTokenizer
from mmeval.infer.task import Task
from mmeval.utils.argparser import parse_args

class TaskRunner(Task):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, args):

        model_path = f"h2oai/{args.model_name_or_path}"

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config.llm_config._attn_implementation = 'flash_attention_2'

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            config=config,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        question = self.parse_input(sample)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(question, sample["media_dir"], None)
        else:
            ori_sample.update(self._score_choices(question, sample["media_dir"], None, sample))

        return ori_sample

    def _generate_response(self, text, image_inputs, video_inputs):
        
        generation_config = dict(max_new_tokens=2048, do_sample=True)

        response = self.model.chat(self.tokenizer, 
                                            image_inputs[0], 
                                            text, 
                                            generation_config, 
                                            history=None, 
                                            return_history=False)
    

        return response

    def _score_choices(self, text, image_inputs, video_inputs, sample):
        raise NotImplementedError("Scoring is not supported.")

    def parse_input(self, sample:dict):
        return sample["prompt"]

    

if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
