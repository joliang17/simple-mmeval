import os, sys, importlib
d = os.path.abspath(os.path.dirname(__file__))
popped = sys.path.pop(0) if sys.path and os.path.abspath(sys.path[0]) == d else None
try:
    m = importlib.import_module('janus')  
    sys.modules['janus'] = m        
finally:
    if popped is not None:
        sys.path.insert(0, popped)

import copy
import torch

from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_gen_kwargs = {
            "max_new_tokens": 512,
            "do_sample": False,
            "use_cache": True
        }
        self.model_kwargs = parse_model_kwargs(args)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        # specify the path to the model
        model_path = args.model_name_or_path
        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.vl_gpt = self.vl_gpt.to(self.dtype).cuda().eval()
        
    def _parse_input(self, sample:dict):
        prompt = sample["prompt"]
        images = sample["media"]

        content = prompt.replace("<image>", "<image_placeholder>\n")

        conversation = [
            {
                "role": "User",
                "content": content,
                "images": images,
            },
            {"role": "Assistant", "content": ""},
        ]

        return conversation

    def _generate_response(self, inputs_embeds, prepare_inputs):
        # run the model to get the response
        outputs = self.vl_gpt.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **self.gen_kwargs
        )

        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

        return answer
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        conversation = self._parse_input(ori_sample)

        # load images and prepare for inputs
        pil_images = ori_sample["media"]
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation, images=pil_images, force_batchify=True
        ).to(self.vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(inputs_embeds, prepare_inputs)
        else:
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
