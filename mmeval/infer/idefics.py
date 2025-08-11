from __future__ import annotations

"""Inference runner for Idefics / Idefics2 / Idefics3 model series.
"""

import copy
import re
from typing import List, Tuple
from PIL import Image

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    IdeficsForVisionText2Text,
    BitsAndBytesConfig,
)

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens


class TaskRunner(Task):
    """Run inference for a single shard/dataset using an Idefics-family model."""

    def __init__(self, args):
        self.args = args  # retain reference for later methods
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Detect instruct models (official naming convention contains "instruct")
        self.is_instruct = "instruct" in args.model_name_or_path.lower()
        
        # Set up default kwargs following Gemma 3 pattern
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 128, "do_sample": False}
        
        # Parse kwargs from args, with fallback to defaults
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        
        super().__init__(args)

    # ---------------------------------------------------------------------
    # Model loading
    # ---------------------------------------------------------------------
    def load_model(self, args):
        """Load the model according to --dtype flag (fp32/fp16/bf16/8bit/4bit)."""

        torch_dtype = None  # float precision for non-quantized loads
        quant_cfg = None    # BitsAndBytesConfig for int-quantized loads

        # Normalize dtype string for easier matching
        dtype_flag = (args.dtype or "auto").lower()

        if dtype_flag in ["auto", "none", "fp16", "float16", "fp32", "float32", "bf16", "bfloat16"]:
            # Pure floating-point paths ----------------------------------
            if dtype_flag in ["auto", "none"]:
                torch_dtype = "auto"
            elif dtype_flag in ["fp16", "float16"]:
                torch_dtype = torch.float16
            elif dtype_flag in ["bf16", "bfloat16"]:
                torch_dtype = torch.bfloat16
            elif dtype_flag in ["fp32", "float32"]:
                torch_dtype = torch.float32
        elif dtype_flag == "8bit":
            quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        elif dtype_flag == "4bit":
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            raise ValueError(f"Unsupported dtype value: {args.dtype}")

        # ----------------------------------------------------------------
        # Detect model family from config
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(args.model_name_or_path)

        if cfg.model_type == "idefics2":
            from transformers import Idefics2ForConditionalGeneration as ModelClass
        elif cfg.model_type == "idefics3":
            from transformers import Idefics3ForConditionalGeneration as ModelClass
        else:  # fallback to original Idefics (v1)
            ModelClass = IdeficsForVisionText2Text

        def _load_model(_torch_dtype, _quant_cfg):
            # Combine parsed model kwargs with specific loading parameters
            load_kwargs = {
                **self.model_kwargs,
                "torch_dtype": _torch_dtype,
                "quantization_config": _quant_cfg,
            }
            return ModelClass.from_pretrained(args.model_name_or_path, **load_kwargs)

        try:
            # First attempt with the chosen setup
            self.model = _load_model(torch_dtype, quant_cfg)
        except RuntimeError as e:
            # Fallback path: If OOM occurs and no quantization was requested, retry with 4-bit
            if ("out of memory" in str(e).lower() or "cuda error" in str(e).lower()) and quant_cfg is None:
                print("[Warning] Initial load ran out of GPU memory; retrying with 4-bit quantization …")
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self.model = _load_model(None, quant_cfg)
            else:
                raise

        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # ---------------------------------------------------------------------
    # Main per-sample entry point
    # ---------------------------------------------------------------------
    def run_sample(self, sample: dict):
        """Run generation or scoring on a single sample from the dataset."""
        output_sample = copy.deepcopy(sample)

        messages = self._build_messages(sample)
        prompt_text = self._apply_chat_template_safe(messages)
        image_inputs, _ = self._extract_media(messages)

        if not self.args.score_target:
            output_sample["response"] = self._generate_response(prompt_text, image_inputs)
        else:
            output_sample.update(self._score_choices(prompt_text, image_inputs, sample))

        return output_sample

    # ------------------------------------------------------------------
    # Generation path
    # ------------------------------------------------------------------
    def _generate_response(self, text: str, image_inputs: List[str]):
        """Generate response with proper stop tokens and bad words filtered."""
        if isinstance(text, list):
            text = " ".join(text)
        
        inputs = self.processor(text=text, images=image_inputs, return_tensors="pt").to(self.device)

        # Common bad words (<image> tokens should not appear in text output)
        bad_words_ids = self.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

        # Combine parsed generation kwargs with model-specific parameters
        gen_kwargs = {
            "bad_words_ids": bad_words_ids,
            **self.gen_kwargs  # Use parsed generation arguments from argparser
        }
        
        # Add instruct-specific parameters if needed
        if self.is_instruct:
            eos_id = self.tokenizer.convert_tokens_to_ids("<end_of_utterance>")
            gen_kwargs["eos_token_id"] = eos_id

        generated = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = generated[:, inputs["input_ids"].shape[-1]:]

        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

        return decoded

    # ------------------------------------------------------------------
    # Scoring path (multiple-choice)
    # ------------------------------------------------------------------
    def _score_choices(self, text: str, image_inputs: List[str], sample: dict):
        """Compute conditional probabilities for each choice and pick the best one."""
        choices = sample["choices"]
        full_prompts = [text + c for c in choices]

        full_encoded = [
            self.processor(text=p, images=image_inputs, return_tensors="pt").to(self.device)
            for p in full_prompts
        ]
        prompt_encoded = self.processor(
            text=text, images=image_inputs, return_tensors="pt"
        ).to(self.device)
        tgt_tokens = target_tokens(self.tokenizer, choices)

        # Use device="auto" to avoid .to(...) which conflicts with accelerate offload hooks
        scorer = IncrementalLMScorer(self.model, device="auto", tokenizer=self.tokenizer)
        scores = scorer.conditional_score(tgt_tokens, full_encoded, prompt_encoded)
        return {
            "score": scores,
            "response": choices[int(np.argmax(scores))],
        }

    # ------------------------------------------------------------------
    # Helper functions
    # ------------------------------------------------------------------
    def _build_messages(self, sample: dict):
        """Convert dataset sample into the chat-template message format required."""
        question: str = sample["prompt"]
        images: List[str] = copy.deepcopy(sample["media"])

        messages = [{"role": "user", "content": []}]

        # Split by placeholders like <image>, <video>, etc.
        for chunk in re.split(r"(<[^>]*>)", question):
            if not chunk.strip():
                continue
            if any(ph in chunk for ph in constants.all):
                # Currently only image placeholders are supported for Idefics
                assert chunk == constants.image, f"Unsupported placeholder {chunk}"
                messages[0]["content"].append({"type": "image", "image": images.pop(0)})
            else:
                messages[0]["content"].append({"type": "text", "text": chunk})
        return messages

    def _extract_media(self, messages) -> Tuple[List[str], List[str]]:
        """Extract image/video file names from built messages for processor feed."""
        image_files: List[str] = []
        video_files: List[str] = []
        for item in messages[0]["content"]:
            if item["type"] == "image":
                image_files.append(item["image"])
            elif item["type"] == "video":
                video_files.append(item["video"])
        return image_files, video_files

    def _apply_chat_template_safe(self, messages):
        """Return prompt text; fallback to naive concatenation if chat template is missing."""
        try:
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (AttributeError, ValueError):
            # Basic fallback
            txt = []
            for item in messages[0]["content"]:
                if item["type"] == "text":
                    txt.append(item["text"])
                elif item["type"] == "image":
                    txt.append(constants.image)
            return f"User: {' '.join(txt)}\nAssistant:"


if __name__ == "__main__":
    TaskRunner(parse_args()).inference_dataset()