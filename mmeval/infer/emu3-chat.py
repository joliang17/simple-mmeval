
import re
import copy
import torch
import numpy as np
import sys
from PIL import Image
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from transformers.generation.configuration_utils import GenerationConfig
from processing_emu3 import Emu3Processor

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens


class TaskRunner(Task):
    def __init__(self, args):
        print('cathy debug: emu3-chat.py, __init__')
        super().__init__(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Add model path to sys.path for custom imports
        sys.path.append(args.model_name_or_path)
    
    def load_model(self, args):
        VQ_HUB = "BAAI/Emu3-VisionTokenier"  # Vision tokenizer is separate
        
        print('cathy debug: emu3-chat.py, load_model')
        print(f"Loading model from: {args.model_name_or_path}")
        
        # Load main chat model
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="cuda:0",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        
        # Load tokenizer with left padding for batch generation
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, 
            trust_remote_code=True, 
            padding_side="left"
        )
        
        # Load image processor and tokenizer
        self.image_processor = AutoImageProcessor.from_pretrained(
            VQ_HUB, 
            trust_remote_code=True
        )
        self.image_tokenizer = AutoModel.from_pretrained(
            VQ_HUB, 
            device_map="cuda:0", 
            trust_remote_code=True
        ).eval()
        
        # Create unified processor
        self.processor = Emu3Processor(
            self.image_processor, 
            self.image_tokenizer, 
            self.tokenizer
        )
        
        # Generation configuration
        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=1024,
        )
        
    def run_sample(self, sample: dict):
        print('cathy debug: emu3-chat.py, run_sample')
        ori_sample = copy.deepcopy(sample)
        
        text, images = self.parse_input(sample)
        
        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(text, images)
        else:
            ori_sample.update(self._score_choices(text, images, sample))

        return ori_sample

    def _generate_response(self, text, image_inputs):
        print('cathy debug: emu3-chat.py, generate_response')
        
        inputs = self.processor(
            text=text,
            image=image_inputs,
            mode='U',
            return_tensors="pt",
            padding="longest",
        )
        
        outputs = self.model.generate(
            inputs.input_ids.to("cuda:0"),
            self.generation_config,
            attention_mask=inputs.attention_mask.to("cuda:0"),
        )
        
        outputs = outputs[:, inputs.input_ids.shape[-1]:]
        print(self.processor.batch_decode(outputs, skip_special_tokens=True)[0])

        return outputs

    def _score_choices(self, text, image_inputs, video_inputs, sample):
        print('cathy debug: emu3-chat.py, score_choices')
        choices = sample.get("choices")
        
        # Prepare prompts with each choice appended
        full_texts = [text + choice for choice in choices]
        
        # Build inputs for all choices
        full_encoded = []
        for full_text in full_texts:
            inputs = self.processor(
                text=full_text,
                image=image_inputs,
                mode='U',
                return_tensors="pt",
                padding="longest",
            ).to(self.device)
            full_encoded.append(inputs)
        
        # Build input for base prompt (without choices)
        prompt_encoded = self.processor(
                text=text,
                image=image_inputs,
                mode='U',
                return_tensors="pt",
                padding="longest",
            ).to(self.device)
        
        # Get target tokens for each choice
        target_toks = target_tokens(self.tokenizer, choices)
        
        # Score using the framework's scorer
        scorer = IncrementalLMScorer(self.model, self.device, tokenizer=self.tokenizer)
        scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)
        
        # Return the choice with highest score
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
