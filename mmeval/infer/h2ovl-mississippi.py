import os
import copy
import torch
from transformers import AutoTokenizer
from mmeval.infer.task import Task
from mmeval.utils.argparser import parse_args

class TaskRunner(Task):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, args):
        model_dir = os.path.abspath("./mmeval/infer/h2ovl_mississippi")
        weight_dir = os.path.abspath("/models/h2ovl_mississippi")
        if "800m" in args.model_name_or_path:
            model_dir = os.path.join(model_dir, "h2o_800m")
            weight_dir = os.path.join(weight_dir, "h2o_800m")
            from mmeval.infer.h2ovl_mississippi.h2o_800m.configuration_h2ovl_chat import H2OVLChatConfig
            from mmeval.infer.h2ovl_mississippi.h2o_800m.modeling_h2ovl_chat import H2OVLChatModel
        else:
            raise ValueError(f"Unsupported model: {args.model_name_or_path}")

        config = H2OVLChatConfig.from_pretrained(model_dir, local_files_only=True)
        config.llm_config._attn_implementation = 'flash_attention_2'

        self.model = H2OVLChatModel.from_pretrained(
            weight_dir,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            config=config,
            low_cpu_mem_usage=True).eval().cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, use_fast=False)
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        question = self.parse_input(sample)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(question, sample["media"], None)
        else:
            ori_sample.update(self._score_choices(question, sample["media"], None, sample))

        return ori_sample

    def _generate_response(self, text, image_inputs, video_inputs):
        
        generation_config = dict(max_new_tokens=2048, do_sample=True)

        response = self.model.chat(self.tokenizer,
                                   image_inputs, 
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
