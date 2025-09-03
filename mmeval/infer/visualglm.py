import re
import copy
import torch
import os
from transformers import AutoTokenizer, AutoConfig
from mmeval.infer.task import Task
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs

from visualglm_6b.modeling_chatglm import ChatGLMForConditionalGenerationWithImage

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        model_dir = os.path.join(os.path.abspath("mmeval/infer"), args.model_name_or_path)
        weight_dir = os.path.join(os.path.abspath("/models"), args.model_name_or_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
        self.config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
        self.model = ChatGLMForConditionalGenerationWithImage.from_pretrained(
            weight_dir,
            local_files_only=True,
            config=self.config,
            **self.model_kwargs).half().cuda()

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        question = self.parse_input(sample)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(question, sample["media"], None)
        else:
            ori_sample.update(self._score_choices(question, sample["media"], None, sample))

        return ori_sample

    def _generate_response(self, text, image_inputs, video_inputs):
        response, history = self.model.chat(self.tokenizer, image_inputs[0], text, history=[])
        return response

    def _score_choices(self, text, image_inputs, video_inputs, sample):
        raise NotImplementedError("Scoring is not supported yet.")

    def parse_input(self, sample:dict):
        question = sample["prompt"]
        # extract placeholder
        placeholders = re.findall(r'<(?:image|video)>', question)
        assert len(placeholders) == 1, f"VisualGLM supports one image, but got {len(placeholder)}"
        placeholder = placeholders[0]

        # remove the placeholder in the question
        question = question.replace(placeholder, "").strip()

        return question



if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()