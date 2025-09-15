from __future__ import annotations

import copy
import re
from typing import List

import torch
from transformers import AutoProcessor, IdeficsForVisionText2Text

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs


class TaskRunner(Task):
    """Inference for Idefics v1 (9B/80B, incl. -instruct).

    Notes
    - Idefics1 does not support `apply_chat_template`; we build prompts manually
      following the official example on the model card.
    - Keep structure consistent with idefics2/3 where possible (init, load, run).
    """

    def __init__(self, args):
        self.args = args
        # For Idefics v1, follow official example: fix to bf16
        self.dtype = torch.bfloat16
        self.is_instruct = "instruct" in args.model_name_or_path.lower()

        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 100, "do_sample": False}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    # -----------------------------
    # Model loading
    # -----------------------------
    def load_model(self, args):
        self.model = (
            IdeficsForVisionText2Text
            .from_pretrained(
                args.model_name_or_path,
                torch_dtype=self.dtype,
                **self.model_kwargs,
            )
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    # -----------------------------
    # Prompts builder (official style)
    # -----------------------------
    def _build_prompts(self, sample: dict) -> List[List[object]]:
        """Build Idefics1 prompts directly from sample using official style.

        - Interleave text chunks and PIL.Images as they appear in the prompt.
        - For -instruct models, prefix first text with "User:", then append
          "<end_of_utterance>" and "\nAssistant:" at the end.
        """
        prompt = sample["prompt"]
        chunks = re.split(r"(<[^>]*>)", prompt)
        media = copy.deepcopy(sample["media"])  # PIL images from dataset loader

        seq: List[object] = []
        first_text = True
        for chunk in chunks:
            if not chunk.strip():
                continue
            if chunk == constants.image:
                seq.append(media.pop(0))
            else:
                text = chunk
                if self.is_instruct and first_text:
                    text = f"User: {text}"
                    first_text = False
                seq.append(text)

        if self.is_instruct:
            seq.append("<end_of_utterance>")
            seq.append("\nAssistant:")

        return [seq]

    # -----------------------------
    # Generation
    # -----------------------------
    def _generate_response(self, inputs, input_len: int):
        tokenizer = self.processor.tokenizer
        bad_words_ids = tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

        gen_kwargs = {"bad_words_ids": bad_words_ids, **self.gen_kwargs}
        if self.is_instruct:
            eos_id = tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
            gen_kwargs["eos_token_id"] = eos_id

        with torch.inference_mode():
            generated = self.model.generate(**inputs, **gen_kwargs)
            # Remove input prefix to return only new tokens
            generated = generated[0][input_len:]

        return self.processor.decode(generated, skip_special_tokens=True)

    # -----------------------------
    # Main per-sample entry
    # -----------------------------
    def run_sample(self, sample: dict):
        ori = copy.deepcopy(sample)
        prompts = self._build_prompts(ori)

        proc_kwargs = {"return_tensors": "pt"}
        if self.is_instruct:
            proc_kwargs["add_end_of_utterance_token"] = False

        # Use official `prompts` argument to ensure images are processed
        inputs = self.processor(prompts=prompts, **proc_kwargs)
        # Follow official example: only move to device, don't recast dtype
        inputs = inputs.to(self.model.device)

        input_len = inputs["input_ids"].shape[-1]

        if not self.args.score_target:
            ori["response"] = self._generate_response(inputs, input_len)
        else:
            # Align with idefics2/3: scoring path not implemented for Idefics1
            pass

        return ori


if __name__ == "__main__":
    args = parse_args()
    TaskRunner(args).inference_dataset()
