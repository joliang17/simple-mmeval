import re
import copy
import torch

from transformers import AutoProcessor, Gemma3ForConditionalGeneration

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

        super().__init__(args)
        
    def load_model(self, args):
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
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
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
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
                messages[1]["content"].append(
                    {
                        "type": "image",
                        "image": media_file
                    }
                )       
            else:
                messages[1]["content"].append(
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
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
