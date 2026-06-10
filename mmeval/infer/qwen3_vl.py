import re
import copy
import os

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
        self.use_vllm = bool(getattr(args, "use_vllm", False))
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 128}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)
        
    def load_model(self, args):
        self.processor = AutoProcessor.from_pretrained(args.model_name_or_path)

        if self.use_vllm:
            os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
            from vllm import LLM

            self.model = None
            self.llm = LLM(
                model=args.model_name_or_path,
                trust_remote_code=True,
                dtype=self.dtype,
                tensor_parallel_size=max(1, torch.cuda.device_count()),
                seed=0,
            )
            return

        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model_name_or_path, 
            dtype=self.dtype, 
            **self.model_kwargs
        )

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
        gen_kwargs = {
            key: value
            for key, value in self.gen_kwargs.items()
            if key != "presence_penalty"
        }
        generated_ids = self.model.generate(**inputs, **gen_kwargs)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text

    def _build_vllm_input(self, user_message):
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
            user_message,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        mm_data = {}
        if images is not None:
            mm_data["image"] = images
        if videos is not None:
            mm_data["video"] = videos

        return {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

    def _build_sampling_params(self):
        from vllm import SamplingParams

        supported_kwargs = {}
        for key, value in self.gen_kwargs.items():
            if key == "max_new_tokens":
                supported_kwargs["max_tokens"] = value
            elif key in {
                "temperature",
                "top_k",
                "top_p",
                "min_p",
                "repetition_penalty",
                "presence_penalty",
                "length_penalty",
            }:
                supported_kwargs[key] = value

        return SamplingParams(**supported_kwargs)

    def _generate_response_vllm(self, user_message):
        output = self.llm.generate(
            [self._build_vllm_input(user_message)],
            sampling_params=self._build_sampling_params(),
        )[0]
        return [output.outputs[0].text]

    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        message = sample["messages"][0]
        
        user_message = self.parse_input(message)

        if self.use_vllm:
            if not self.args.score_target:
                response = self._generate_response_vllm(user_message)
                ori_sample["messages"].append({"role": "assistant", "response": response})
            return ori_sample
        
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
