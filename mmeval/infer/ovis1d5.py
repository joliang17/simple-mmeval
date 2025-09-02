import re
import copy
import torch

from transformers import AutoModelForCausalLM

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_gen_kwargs = {
            "max_new_tokens": 1024, 
            "do_sample": False,
            "top_p": None,
            "top_k": None,
            "temperature": None,
            "repetition_penalty": None,
            "use_cache": True
        }
        self.model_kwargs = parse_model_kwargs(args)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)
        
    def load_model(self, args):
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                    torch_dtype=self.dtype,
                                                    multimodal_max_length=8192,
                                                    trust_remote_code=True,
                                                    **self.model_kwargs).cuda()
        self.text_tokenizer = self.model.get_text_tokenizer()
        self.visual_tokenizer = self.model.get_visual_tokenizer()
        self.conversation_formatter = self.model.get_conversation_formatter()
        
    def _parse_input(self, sample:dict):
        prompt = sample["prompt"]
        query = prompt.replace("<image>", "<image>\n")

        return query

    def _generate_response(self, input_ids, attention_mask, pixel_values):
        # generate output
        with torch.inference_mode():
            self.gen_kwargs["eos_token_id"] = self.model.generation_config.eos_token_id
            self.gen_kwargs["pad_token_id"] = self.text_tokenizer.pad_token_id

            output_ids = self.model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **self.gen_kwargs)[0]
            output = self.text_tokenizer.decode(output_ids, skip_special_tokens=True)

        return output
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)

        query = self._parse_input(ori_sample)
        prompt, input_ids = self.conversation_formatter.format_query(query)
        input_ids = torch.unsqueeze(input_ids, dim=0).to(device=self.model.device)
        attention_mask = torch.ne(input_ids, self.text_tokenizer.pad_token_id).to(device=self.model.device)
        
        image = ori_sample['media'][0]
        pixel_values = [self.visual_tokenizer.preprocess_image(image).to(
            dtype=self.visual_tokenizer.dtype, device=self.visual_tokenizer.device)]

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(input_ids, attention_mask, pixel_values)
        else:
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
