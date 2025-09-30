
import re
import copy
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens


class TaskRunner(Task):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, args):
        print(f"Loading model from: {args.model_name_or_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device).eval()

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        
        text, images = self.parse_input(sample)
        
        print(f"sample text: {text}")
        print(f"sample images: {images}")
        
        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(text, images)
        else:
            ori_sample.update(self._score_choices(text, images, sample))

        return ori_sample

    def _generate_response(self, text, image_inputs):
        inputs = self.model.build_input_ids(
            text=[text],
            tokenizer=self.tokenizer,
            image=image_inputs
        )
        inputs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image=inputs["image"].to(torch.bfloat16),
                max_new_tokens=64,
                length_penalty=-1, 
            )
        
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return output_text

    def _score_choices(self, text, image_inputs, video_inputs, sample):
        choices = sample.get("choices")
        
        full_texts = [text + choice for choice in choices]
        full_encoded = []
        for full_text in full_texts:
            encoded = self.model.build_input_ids(
                text=[full_text],
                tokenizer=self.tokenizer,
                image=image_inputs
            ).to(self.device)
            full_encoded.append(encoded)
        
        prompt_encoded = self.model.build_input_ids(
            text=[text],
            tokenizer=self.tokenizer,
            image=image_inputs
        ).to(self.device)
        
        target_toks = target_tokens(self.tokenizer, choices)
        scorer = IncrementalLMScorer(self.model, self.device, tokenizer=self.tokenizer)
        scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)

        return {
            "score": scores,
            "response": choices[np.argmax(scores)]
        }

    def parse_input(self, sample:dict):
        question = sample["prompt"]
        media_files = copy.deepcopy(sample['media'])
        
        if "<video>" in question:
            print("Warning: Emu2-Chat may not support video inputs.")
            return
        
        question = question.replace("<image>", "[<IMG_PLH>]")
        
        images = []
        for media_path in media_files:
            if isinstance(media_path, str):
                img = Image.open(media_path).convert('RGB')
            else:
                img = media_path   
            images.append(img)

        placeholder_count = question.count("[<IMG_PLH>]")
        if placeholder_count != len(images):
            print(f"Warning: Placeholder count ({placeholder_count}) != image count ({len(images)})")
        
        return question, images
    

if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
