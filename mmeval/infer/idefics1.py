from __future__ import annotations

"""Inference runner for Idefics1 model."""

import copy
import re
from typing import List, Tuple

import torch
from transformers import AutoProcessor, IdeficsForVisionText2Text, BitsAndBytesConfig

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens


class TaskRunner(Task):
    """Run inference for a single shard/dataset using Idefics1 model."""

    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Detect instruct models (official naming convention contains "instruct")
        self.is_instruct = "instruct" in args.model_name_or_path.lower()
        
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 100, "do_sample": False}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)
        
    def load_model(self, args):
        """Load Idefics1 model and processor."""
        # Handle different dtype configurations
        dtype_flag = (getattr(args, 'dtype', None) or "auto").lower()
        
        if dtype_flag == "4bit":
            # Use 4-bit quantization to save memory
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model_kwargs = {**self.model_kwargs, "quantization_config": quantization_config}
        elif dtype_flag == "8bit":
            # Use 8-bit quantization
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            model_kwargs = {**self.model_kwargs, "quantization_config": quantization_config}
        else:
            model_kwargs = self.model_kwargs
            
        try:
            self.model = IdeficsForVisionText2Text.from_pretrained(
                args.model_name_or_path, **model_kwargs
            ).eval()
        except RuntimeError as e:
            # Fallback to 4-bit if OOM occurs
            if "out of memory" in str(e).lower() or "cuda error" in str(e).lower():
                print("[Warning] Initial load ran out of GPU memory; retrying with 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
                fallback_kwargs = {**self.model_kwargs, "quantization_config": quantization_config}
                self.model = IdeficsForVisionText2Text.from_pretrained(
                    args.model_name_or_path, **fallback_kwargs
                ).eval()
            else:
                raise
                
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)
        
    def _parse_input(self, sample: dict):
        """Parse input sample into messages format, similar to gemma3 but adapted for idefics1."""
        prompt = sample["prompt"]
        # placeholder <>, can be image, video, audio, etc.
        q_chunks = re.split(r'(<[^>]*>)', prompt)
        media = copy.deepcopy(sample['media'])

        # For idefics1, we use a simpler message structure without system role
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

    def _extract_media(self, messages) -> Tuple[List, List]:
        """Extract image/video file names from built messages for processor feed."""
        image_files: List = []
        video_files: List = []
        for item in messages[0]["content"]:
            if item["type"] == "image":
                image_files.append(item["image"])
            elif item["type"] == "video":
                video_files.append(item["video"])
        return image_files, video_files

    def _apply_chat_template_safe(self, messages):
        """Return prompt text; fallback to naive concatenation since idefics1 doesn't support chat template."""
        try:
            # Try using processor's chat template (likely to fail for idefics1)
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (AttributeError, ValueError):
            # Basic fallback for idefics1 - construct simple format
            txt = []
            for item in messages[0]["content"]:
                if item["type"] == "text":
                    txt.append(item["text"])
                elif item["type"] == "image":
                    txt.append(constants.image)
            return f"User: {' '.join(txt)}\nAssistant:"

    def _generate_response(self, text: str, image_inputs: List):
        """Generate response using the correct processor format for idefics1."""
        if isinstance(text, list):
            text = " ".join(text)
        
        # Use correct processor call for idefics1
        inputs = self.processor(text=text, images=image_inputs, return_tensors="pt").to(self.device)

        # Generate response
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **self.gen_kwargs)
            new_tokens = generated[:, inputs["input_ids"].shape[-1]:]
            
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        return decoded
    
    def run_sample(self, sample: dict):
        """Run generation or scoring on a single sample from the dataset."""
        ori_sample = copy.deepcopy(sample)
        messages = self._parse_input(ori_sample)

        # Since idefics1 doesn't support chat template, we need to handle this differently
        prompt_text = self._apply_chat_template_safe(messages)
        image_inputs, _ = self._extract_media(messages)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(prompt_text, image_inputs)
        else:
            # Scoring not implemented for idefics1 yet
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset() 