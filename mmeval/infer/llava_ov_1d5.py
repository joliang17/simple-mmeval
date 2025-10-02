import re
import copy
import torch
import numpy as np
import transformers
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

from qwen_vl_utils import process_vision_info

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 100, "do_sample": False}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        super().__init__(args)
    
    def load_model(self, args):
        self.model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **self.model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages = self.parse_input(sample)
        
    
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(text, image_inputs, video_inputs)
        else:
            ori_sample.update(self._score_choices(text, image_inputs, video_inputs, sample))

        return ori_sample

    def _generate_response(self, text, image_inputs, video_inputs):
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if "second_per_grid_ts" in inputs:
            inputs.pop("second_per_grid_ts")  # TODO: need to check this
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, **self.gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return output_text

    def _score_choices(self, text, image_inputs, video_inputs, sample):
        contents = sample.get("choices")
        full = [text + content for content in contents]

        full_encoded = [self.processor(text=i, images=image_inputs, videos=video_inputs, return_tensors="pt").to(self.model.device) for i in full]
        prompt_encoded = self.processor(text=text, images=image_inputs, videos=video_inputs, return_tensors="pt").to(self.model.device)
        target_toks = target_tokens(self.tokenizer, contents)

        scorer = IncrementalLMScorer(self.model, self.model.device, tokenizer=self.tokenizer)
        scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)
        
        return {
            "score": scores,
            "response": contents[np.argmax(scores)]
        }


    def parse_input(self, sample:dict):
        question = sample["prompt"]
        # placeholder <>, can be image, video, etc.
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        images = copy.deepcopy(sample['media'])

        messages = [
            {
                "role": "user",
                "content": []
            }
        ]

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            
            if any(p in chunk for p in constants.all):
                
                assert chunk == constants.image or chunk == constants.video, f"Unsupported placeholder {chunk}"

                media_file = images.pop(0)
                if chunk == constants.image:
                    messages[0]["content"].append(
                        {
                            "type": "image",
                            "image": media_file
                        }
                    )       
                elif chunk == constants.video:
                    messages[0]["content"].append(
                        {
                            "type": "video",
                            "video": media_file
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

    

if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
