import re
import copy
import torch
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer

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
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        super().__init__(args)
        
    def load_model(self, args):
        self.model = AutoModelForVision2Seq.from_pretrained(
            args.model_name_or_path, **self.model_kwargs
        ).eval()
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)
        
    def _parse_input(self, sample:dict):
        prompt = sample["prompt"]
        # placeholder <>, can be image, video, audio, etc.
        q_chunks = re.split(r'(<(?:image|video)>)', prompt)
        media = copy.deepcopy(sample['media'])

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
                media_file = media.pop(0)
                messages[0]["content"].append(
                    {
                        "type": "image",
                        "image": media_file
                    }
                )       
            else:
                messages[0]["content"].append(
                    {
                        "type": "text",
                        "text": chunk
                    }
                )

        return messages

    def _generate_response(self, inputs, input_len):
        with torch.inference_mode():
            generation = self.model.generate(**inputs, **self.gen_kwargs)
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)

        return decoded
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages = self._parse_input(ori_sample)

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=self.dtype)

        input_len = inputs["input_ids"].shape[-1]

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(inputs, input_len)
        else:
            ori_sample.update(self._score_choices(messages, "image", ori_sample))

        return ori_sample
    
    def _score_choices(self, messages, modality, sample):
        media = copy.deepcopy(sample['media'])
        contents = sample.get("choices")
        if modality == "image":
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full = [text + content for content in contents]
            full_encoded = [self.processor(text=i, images=media, return_tensors="pt").to(self.device) for i in full]
            prompt_encoded = self.processor(text=text, images=media, return_tensors="pt").to(self.device)

        target_toks = target_tokens(self.tokenizer, contents)

        scorer = IncrementalLMScorer(self.model, self.device, tokenizer=self.tokenizer)
        scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)
        
        return {
            "score": scores,
            "response": contents[np.argmax(scores)]
        }

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
