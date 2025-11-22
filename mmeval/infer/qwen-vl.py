import re
import copy
import torch
import numpy as np
import tempfile
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
        torch.manual_seed(1234)
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
            device_map="cuda",
            trust_remote_code=True
        ).eval()
        
        # Not needed if using transformers>=4.32.0
        # self.model.generation_config = GenerationConfig.from_pretrained(
        #     args.model_name_or_path, 
        #     trust_remote_code=True
        # )
        
    def run_sample(self, sample: dict):
        """Process a single sample through Qwen-VL"""
        
        ori_sample = copy.deepcopy(sample)
        
        query_list = self.parse_input(sample)
        
        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(query_list)
        else:
            ori_sample.update(self._score_choices(query_list, sample))

        return ori_sample

    def _generate_response(self, query_list):
        print('cathy debug, generate_response')
        """Generate response using Qwen-VL's API"""

        # Convert list format to query string
        query = self.tokenizer.from_list_format(query_list)
        print(f'query: {query}')
        
        # Tokenize the query
        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.device)
        
        # Generate response
        pred = self.model.generate(**inputs)
        pred_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, pred)
        ]
        
        # Decode response
        response = self.tokenizer.decode(pred_trimmed[0], skip_special_tokens=False)
        
        # Extract only the generated response (after the input query)
        # The response format includes special tokens, we need to clean it
        if '<|endoftext|>' in response:
            response = response.split('<|endoftext|>')[0]
        
        print(f'response: {response}')
        
        return response

    def _score_choices(self, query_list, sample):
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
        
        # Build query list in Qwen-VL format
        query_list = []
        image_counter = 0
        
        for chunk in q_chunks:
            print(f'chunk {chunk}')
            if not chunk.strip():
                continue
            
            if any(p in chunk for p in constants.all):
                assert chunk == constants.image, f"Unsupported placeholder {chunk}"
                media_file = media_files.pop(0)
                
                if isinstance(media_file, Image.Image):
                    # Save PIL Image to temp file and use that path
                    temp_path = os.path.join(self.temp_dir, f"temp_image_{image_counter}.jpg")
                    media_file.save(temp_path)
                    print(f"Converted PIL Image to temp file: {temp_path}")
                    media_file = temp_path
                    image_counter += 1
                elif isinstance(media_file, str):
                    # Already a string path, use directly
                    print(f"Using image path directly: {media_file}")
                query_list.append({'image': media_file})
                
                print('appended image')
            else:
                query_list.append({'text': chunk})
                print('appended text')
        
        if media_files:
            print(f"Warning: {len(media_files)} media files were not used")
        
        return query_list

# Required: Main execution block
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()