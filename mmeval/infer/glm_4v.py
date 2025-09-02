import re
import copy
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

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

    def load_model(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            **self.model_kwargs
        ).to(self.device).eval()
        
    def _parse_input(self, sample:dict):
        prompt = sample["prompt"]
        query = prompt.replace("<image>", "")

        return query

    def _generate_response(self, inputs):
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]

        return self.tokenizer.decode(outputs[0])
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        query = self._parse_input(ori_sample)
        image = ori_sample['media'][0]
        
        inputs = self.tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                            add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                            return_dict=True)  # chat mode

        inputs = inputs.to(self.device)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(inputs)
        else:
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
