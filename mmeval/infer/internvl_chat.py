import re
import copy

import math
import torch
from transformers import AutoTokenizer, AutoModel, CLIPImageProcessor
from decord import VideoReader, cpu
from PIL import Image
import numpy as np

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {'InternVL-Chat-V1-1': 40, 'InternVL-Chat-V1-2': 60, 'InternVL-Chat-V1-2-Plus': 60}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    image_processor = CLIPImageProcessor.from_pretrained(args.model_name_or_path)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB').resize((448, 448))
        pixel_values = image_processor(images=img, return_tensors='pt').pixel_values
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)

    return pixel_values_list, num_patches_list

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.num_segments = 8
        self.default_model_kwargs = {"device_map": split_model(re.split(r'/', args.model_name_or_path)[-1]), "low_cpu_mem_usage": True}
        # self.default_gen_kwargs = {"max_new_tokens": 1024, "do_sample": True}
        self.default_gen_kwargs = {"num_beams": 1, "top_k": 50, "top_p": 0.9, "sample": False, "max_new_tokens": 20}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        self.model = AutoModel.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.dtype,
            load_in_8bit=False,
            use_flash_attn=True,
            trust_remote_code=True,
            **self.model_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
        
    def _parse_input(self, sample:dict):
        question = sample["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)
        question = question.replace("<image>", "<image>\n")
        question = question.replace("<video>", ''.join([f'Frame{i+1}: <image>\n' for i in range(self.num_segments)]))

        image_processor = CLIPImageProcessor.from_pretrained(args.model_name_or_path)
        media_list = copy.deepcopy(sample['media'])
        pixel_values_list = []
        num_patches_list = []
        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                image = media_list.pop(0)
                resized_image = image.resize((448, 448), resample=Image.Resampling.LANCZOS)
                image_pixel_values = image_processor(images=resized_image, return_tensors='pt').pixel_values
                pixel_values_list.append(image_pixel_values)
                num_patches_list.append(image_pixel_values.size(0))
            elif chunk == constants.video:
                video = media_list.pop(0)
                video_pixel_values_list, video_num_patches_list = load_video(video, num_segments=self.num_segments)
                pixel_values_list.extend(video_pixel_values_list)
                num_patches_list.extend(video_num_patches_list)

        pixel_values = torch.cat(pixel_values_list, dim=0).to(self.dtype).to(self.device)  
                
        return question, pixel_values, num_patches_list

    def _generate_response(self, question,pixel_values, num_patches_list):
        response = self.model.chat(self.tokenizer, pixel_values, question, self.gen_kwargs,
                                   num_patches_list=num_patches_list, history=None, return_history=False)

        return response
    
    def run_sample(self, sample: dict):
        ori_sample = copy.deepcopy(sample)
        question, pixel_values, num_patches_list = self._parse_input(ori_sample)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(question, pixel_values, num_patches_list)
        else:
            pass

        return ori_sample

    
if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
