import re
import copy
from typing import Dict, List, Tuple

import torch
from transformers import AutoProcessor, AutoModel

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {
            "max_new_tokens": 200,
            "do_sample": True,
            "num_beams": 1,
            "min_length": 1,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
            "length_penalty": 1,
            "temperature": 1.0,
            "stop_criteria_keywords": None,
            "print_res": False,
        }
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        print(args.model_name_or_path)
        self.model = (
            AutoModel.from_pretrained(
                args.model_name_or_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **self.model_kwargs,
            )
            .eval()
            .cuda()
        )
        self.generation_config = dict(max_new_tokens=1024, do_sample=True)
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    def _parse_input(self, sample: Dict) -> Tuple[str, List[str]]:
        prompt: str = sample["prompt"]
        media_list = copy.deepcopy(sample.get("media", []))

        chunks = re.split(r"(<[^>]*>)", prompt)
        text_parts = []
        ordered_videos = []
        for chunk in chunks:
            if not chunk or not chunk.strip():
                continue
            if chunk == constants.video:
                # Consume one media path if available and record order
                if media_list:
                    ordered_videos.append(media_list.pop(0))
                # Do not add the placeholder to text
                continue
            text_parts.append(chunk)

        text = "".join(text_parts).strip()
        return text, ordered_videos

    def _generate_response(self, inputs) -> str:
        with torch.inference_mode():
            generation = self.model.generate(**inputs, **self.gen_kwargs, media_type="video")
            output_text = self.processor.batch_decode(
                generation,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            split_tag = "ASSISTANT:"
            sep=[" ","</s>"]
            ending = sep[1]
            output_text = output_text.split(split_tag)[-1]
            output_text = output_text.removesuffix(ending).strip()
            return output_text

    def _score_choices(self, text, image_inputs, video_inputs, sample):
        # Not implemented for now
        pass

    def run_sample(self, sample: Dict):
        ori_sample = copy.deepcopy(sample)
        text, videos = self._parse_input(ori_sample)

        inputs = (
            self.processor(text=text, videos=videos, return_tensors="pt")
            .to(self.model.device, dtype=self.dtype)
        )

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(inputs)
        else:
            # Scoring path not implemented
            ori_sample["response"] = ""

        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
