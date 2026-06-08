import re
import copy

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(args, "dtype") or "auto"
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 128}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)
        
    def load_model(self, args):
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model_name_or_path, 
            dtype=self.dtype, 
            **self.model_kwargs
        )
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)

    def parse_input(self, message):
        question = message["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        media_list = message.get('media', [])

        messages = [
            {
                "role": "user",
                "content": []
            }
        ]

        media_idx = 0
        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                media = media_list[media_idx]
                media_idx += 1
                messages[0]["content"].append(
                    {
                        "type": "image",
                        "image": media,
                        "min_pixels": 4 * 32 * 32,
                        "max_pixels": 256 * 32 * 32,
                    }
                )       
            elif chunk == constants.video:
                media = media_list[media_idx]
                media_idx += 1
                messages[0]["content"].append(
                    {
                        "type": "video",
                        "video": media,
                        "min_pixels": 4 * 32 * 32,
                        "max_pixels": 256 * 32 * 32,
                        "total_pixels": 20480 * 32 * 32,
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

    def _generate_response(self, inputs):
        generated_ids = self.model.generate(**inputs, **self.gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        message = sample["messages"][0]
        
        user_message = self.parse_input(message)
        
        text = self.processor.apply_chat_template(
            user_message,
            tokenize=False,
            add_generation_prompt=True,
            **(
                {"enable_thinking": self.args.enable_thinking}
                if self.args.enable_thinking is not None
                else {}
            ),
        )

        images, videos, video_kwargs = process_vision_info(
            user_message, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
        )

        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None

        inputs = self.processor(
            text=text, 
            images=images, 
            videos=videos, 
            video_metadata=video_metadatas,
            return_tensors="pt", 
            do_resize=False, 
            **video_kwargs
        )
        inputs = inputs.to(self.model.device)

        if not self.args.score_target:
            response = self._generate_response(inputs)
            ori_sample["messages"].append({"role": "assistant", "response": response})

        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()

