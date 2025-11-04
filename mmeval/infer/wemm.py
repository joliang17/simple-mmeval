import re
import copy
import torch

from transformers import AutoProcessor, AutoModel

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype", None) or torch.bfloat16
        self.default_model_kwargs = {"torch_dtype": self.dtype}
        self.default_gen_kwargs = {"max_new_tokens": 100, "do_sample": False}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)
        
    def load_model(self, args):
        self.model = AutoModel.from_pretrained(
            args.model_name_or_path, trust_remote_code=True,  **self.model_kwargs
        ).eval().to(device="cuda", dtype=self.dtype)
        
    def _parse_input(self, sample):
        prompt = sample["prompt"]
        # placeholder <>, can be image, video, audio, etc.
        q_chunks = re.split(r'(<(?:image|video)>)', prompt)
        media = copy.deepcopy(sample['media'])

        if len(media) > 1:
            raise ValueError("WEMM only supports one image input")
        
        image = media[0]

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                pass  
            elif chunk == constants.video:
                raise ValueError("WEMM only supports image input")
            else:
                query = chunk

        return image, query

    def _generate_response(self, image, query):
        pred = self.model.mm_generate(image, query)

        return pred
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        image, query = self._parse_input(ori_sample)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(image, query)
        else:
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
