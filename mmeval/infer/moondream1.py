"""
Moondream1 is a VLM that can take in single image and text.
https://huggingface.co/vikhyatk/moondream1
"""
import re
import copy
import torch

from transformers import AutoModelForCausalLM, CodeGenTokenizerFast as Tokenizer
from PIL import Image

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {} # Moondream1 does not support device_map = "auto"
        self.default_gen_kwargs = {"max_new_tokens": 100, "do_sample": False}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        super().__init__(args)
    
    def load_model(self, args):
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, 
            trust_remote_code=True,
            **self.model_kwargs
        )
        # Move model to GPU since Moondream1 doesn't support device_map="auto"
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.tokenizer = Tokenizer.from_pretrained(args.model_name_or_path)
    
    def _parse_input(self, sample: dict):
        prompt = sample["prompt"]
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

        # Extract prompt and image_path from messages
        parsed_prompt = ""
        image_path = None
        for content in messages[0]["content"]:
            if content["type"] == "text":
                parsed_prompt += content["text"]
            elif content["type"] == "image":
                image_path = content["image"]

        return parsed_prompt, image_path

    def _generate_response(self, text, image):
        if image:
            if isinstance(image, str):
                image = Image.open(image)
            with torch.inference_mode():
                enc_image = self.model.encode_image(image)
                output = self.model.answer_question(enc_image, text, self.tokenizer)
            return output
        else:
            raise ValueError("Moondream1 requires an image input")

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        prompt, image_path = self._parse_input(ori_sample)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(prompt, image_path)
        else:
            ori_sample.update(self._score_choices(prompt, image_path, sample))

        return ori_sample

    def _score_choices(self, text, image_path, sample):
        pass

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()