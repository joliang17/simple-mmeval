import re
import copy
import torch
import numpy as np
from PIL import Image
import av
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor, AutoTokenizer

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.

    Args:
        container (av.container.input.InputContainer): PyAV container.
        indices (List[int]): List of frame indices to decode.

    Returns:
        np.ndarray: np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 100, "do_sample": False}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(args)
    
    def load_model(self, args):
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(args.model_name_or_path, **self.model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        messages, modality = self.parse_input(sample)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(messages, modality)
        else:
            ori_sample.update(self._score_choices(messages, modality, sample['media'], sample))

        return ori_sample

    def _generate_response(self, messages, modality):
        if modality == "image":
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            ).to(self.device, torch.float16)
        elif modality == "video":
            inputs = self.processor.apply_chat_template(
                messages, num_frames=8, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            ).to(self.device, torch.float16)
        
        generated_ids = self.model.generate(**inputs, **self.gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        return output_text

    def _score_choices(self, messages, modality, media, sample):
        contents = sample.get("choices")
        if modality == "image":
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            full = [text + content for content in contents]
            full_encoded = [self.processor(text=i, images=media, return_tensors="pt").to(self.device) for i in full]
            prompt_encoded = self.processor(text=text, images=media, return_tensors="pt").to(self.device)
        elif modality == "video":
            text = self.processor.apply_chat_template(
                messages, num_frames=8, tokenize=False, add_generation_prompt=True
            )
            full = [text + content for content in contents]
            container = av.open(media[0])

            # sample uniformly 8 frames from the video
            total_frames = container.streams.video[0].frames
            indices = np.arange(0, total_frames, total_frames / 8).astype(int)
            clip = read_video_pyav(container, indices)
            full_encoded = [self.processor(text=i, videos=clip, return_tensors="pt").to(self.device) for i in full]
            prompt_encoded = self.processor(text=text, videos=clip, return_tensors="pt").to(self.device)

        target_toks = target_tokens(self.tokenizer, contents)

        scorer = IncrementalLMScorer(self.model, self.device, tokenizer=self.tokenizer)
        scores = scorer.conditional_score(target_toks, full_encoded, prompt_encoded)
        
        return {
            "score": scores,
            "response": contents[np.argmax(scores)]
        }


    def parse_input(self, sample:dict):
        question = sample["prompt"]
        # placeholder <>, can be image, video, audio, etc.
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

                if chunk == constants.image:
                    modality = "image"
                    messages[0]["content"].append(
                    {
                        "type": modality,
                        modality: images.pop(0)
                    }
                )      
                elif chunk == constants.video:
                    modality = "video"
                    messages[0]["content"].append(
                    {
                        "type": modality,
                        "path": images.pop(0)
                    }
                )

            else:
                messages[0]["content"].append(
                    {
                        "type": "text",
                        "text": chunk
                    }
                )

        return messages, modality

if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
