import re
import copy
import torch
import transformers
import numpy as np
import tempfile
import warnings
import os

from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temp_dir = tempfile.mkdtemp(prefix="qwenvl_images_")
        
    def load_model(self, args):
        """Load Qwen-VL model and tokenizer"""
        print(f"Loading model from: {args.model_name_or_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float16,  # float32 for cpu
            device_map="auto",
            trust_remote_code=True,
        )
        
    def run_sample(self, sample: dict):
        """Process a single sample through Qwen-VL"""
        
        ori_sample = copy.deepcopy(sample)
        
        text, image_inputs = self.parse_input(sample)
        
        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(text, image_inputs)
        else:
            ori_sample.update(self._score_choices(text, image_inputs, sample))

        return ori_sample

    def _generate_response(self, text, image_inputs):
        print('cathy debug, generate_response')
        """Generate response using Qwen-VL's API"""
        
        print(f'text: {text}')
        input_ids = self.model.text_process(text, self.tokenizer).to(self.device)

        print(f'image list: {image_inputs}')
        image_tensor = self.model.image_process(image_inputs).to(dtype=self.model.dtype, device=self.device)
        
        
        # Generate response
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=2048,
                use_cache=True,
                eos_token_id=[
                    self.tokenizer.eos_token_id,
                    self.tokenizer.convert_tokens_to_ids(["<|eot_id|>"])[0],
                ],
            )
    
        output_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)
        ]
        
        # Decode response
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        response_from_trimmed = self.tokenizer.batch_decode(output_ids_trimmed, skip_special_tokens=True)[0].strip()
        
        # Extract only the generated response (after the input query)
        # The response format includes special tokens, we need to clean it
        if '<|endoftext|>' in response:
            response = response.split('<|endoftext|>')[0]
        
        print(f'response: {response}')
        print(f'response_from_trimmed: {response_from_trimmed}')
        
        return response

    def _score_choices(self, text, image_inputs, sample):
        pass
    
    def parse_input(self, sample: dict):
        print('cathy debug, parse_input')
        """
        Convert MMEval format to Qwen-VL format.
        
        MMEval format:
        - sample["prompt"]: Text with <image>, <video> placeholders
        - sample["media"]: List of media file paths
        
        Qwen-VL format:
        - List of dicts: [{'image': path}, {'text': prompt}, ...]
        """
        question = sample["prompt"]
        media_files = copy.deepcopy(sample['media'])
        print('media file type: ', type(media_files[0]))
        
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        
        processed_question = ""
        images = []
        
        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if any(p in chunk for p in constants.all):
                assert chunk == constants.image, f"Unsupported placeholder {chunk}"
                media_file = media_files.pop(0)
                if isinstance(media_file, str):
                    media_file = Image.open(media_file)
                images.append(media_file)
            else:
                processed_question += chunk
        
        print(f'question: {processed_question}')
        print(f'images: {images}')
        
        return processed_question.strip(), images

# Required: Main execution block
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()