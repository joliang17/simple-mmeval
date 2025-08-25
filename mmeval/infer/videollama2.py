import re
import copy
import torch
from transformers import BatchEncoding
import numpy as np

from mmeval.infer.videollama2 import model_init, mm_infer
from mmeval.infer.videollama2.utils import disable_torch_init

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens
from mmeval.infer.videollama2.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN
from mmeval.infer.videollama2.mm_utils import tokenizer_multimodal_token

class TaskRunner(Task):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, args):
        self.model, self.processor, self.tokenizer = model_init(args.model_name_or_path)

    def run_sample(self, sample:dict):

        ori_sample = copy.deepcopy(sample)
        question, modality = self.parse_input(sample)

        image_file = sample["media"][0]

        media_tensor = self.processor[modality](image_file)

        if not self.args.score_target:
            output_text = mm_infer(media_tensor, question, model=self.model, tokenizer=self.tokenizer, do_sample=False, modal=modality).strip()
            ori_sample["response"] = output_text

        else:
            ori_sample.update(self._score_choices(ori_sample, modality, question, media_tensor))
        
        return ori_sample
    
    def _score_choices(self, sample, modality, question, media_tensor):
        contents = sample.get("choices")

        target_toks = target_tokens(self.tokenizer, contents)
        if modality == 'image':
            modal_token = DEFAULT_IMAGE_TOKEN
        elif modality == 'video':
            modal_token = DEFAULT_VIDEO_TOKEN
        elif modality == 'text':
            modal_token = ''
        
        if modality == 'text':
            tensor = None
        else:
            tensor = media_tensor.half().cuda()
            tensor = [(tensor, modality)]
        
    
        message = [{'role': 'user', 'content': modal_token + '\n' + question}]
        if self.model.config.model_type in ['videollama2', 'videollama2_mistral', 'videollama2_mixtral']:
            system_message = [
                {'role': 'system', 'content': (
                """<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature."""
                """\n"""
                """If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>""")
                }
            ]
        else:
            system_message = []

        message = system_message + message
        prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        _, _, _, prompt_encoded, _ = self.model.prepare_inputs_labels_for_multimodal(
            input_ids=tokenizer_multimodal_token(prompt, self.tokenizer, modal_token, return_tensors='pt').unsqueeze(0).to(self.device),
            images=tensor,
            attention_mask=None,
            labels=None,
            past_key_values=None
        )
        full = [tokenizer_multimodal_token(prompt+content, self.tokenizer, modal_token, return_tensors='pt').unsqueeze(0).to(self.device) for content in contents]  # full conversation for each choice
        full_encoded = []
        for i in full:
            _, _, _, tmp, _ = self.model.prepare_inputs_labels_for_multimodal(input_ids=i, images=tensor, attention_mask=None, labels=None, past_key_values=None)
            full_encoded.append(BatchEncoding({"inputs_embeds": tmp,}))

        scorer = IncrementalLMScorer(self.model, self.device, tokenizer=self.tokenizer)
        scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)  # inputs are used to truncate/locate the prompt and choices' contents

        return {
            "score": scores,
            "response": contents[np.argmax(scores)]
        }
    
    def parse_input(self, sample:dict):
        question = sample["prompt"]
        # extract placeholder
        placeholders = re.findall(r'<(?:image|video)>', question)
        assert len(placeholders) == 1, f"VideoLLaMA2 supports one image or video, but got {len(placeholder)}"
        
        placeholder = placeholders[0]
        modality = "image" if placeholder == constants.image else "video"

        # remove the placeholder in the question
        question = question.replace(placeholder, "").strip()

        return question, modality
        

    

if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()

        
