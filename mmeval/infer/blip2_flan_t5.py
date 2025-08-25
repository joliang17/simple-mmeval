import re
import copy
import torch
import numpy as np
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 100, "do_sample": False}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(args)
        
    def load_model(self, args):
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            args.model_name_or_path, **self.model_kwargs
        )
        self.processor = Blip2Processor.from_pretrained(args.model_name_or_path)

    def _parse_input(self, sample: dict):
        question = sample["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        images = copy.deepcopy(sample['media'])
        
        processed_question = ""
        image = None
        
        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            
            if any(p in chunk for p in constants.all):
                assert chunk == constants.image, f"BLIP2 only supports image input, got {chunk}"
                
                media_file = images.pop(0)
                if isinstance(media_file, str):
                    image = Image.open(media_file).convert('RGB')
                else:
                    image = media_file
                    
            else:
                processed_question += chunk
        
        return image, processed_question.strip()

    def _generate_response(self, image, question):
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**inputs, **self.gen_kwargs)
        output_text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return output_text
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        image, question = self._parse_input(ori_sample)
        
        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(image, question)
        else:
            ori_sample.update(self._score_choices(image, question, sample))

        return ori_sample

    def _score_choices(self, image, question, sample):
        pass


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
