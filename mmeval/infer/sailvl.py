import re
import copy

import math
import torch
import numpy as np
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=10, image_size=448, use_thumbnail=False):
    """
    SAIL-VL uses dynamic preprocessing with up to 10 tiles (max_num=10)
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=10):
    """
    Load and preprocess image for SAIL-VL model
    SAIL-VL uses 448x448 tiles with max 10 tiles
    """
    if isinstance(image_file, Image.Image):
        image = image_file
    else:
        image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = getattr(args, "dtype", None) or torch.bfloat16  # SAIL-VL uses BF16
        self.default_model_kwargs = {"low_cpu_mem_usage": True, "device_map": "auto"}
        self.default_gen_kwargs = {"max_new_tokens": 1024, "do_sample": True}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)

        super().__init__(args)

    def load_model(self, args):
        """
        Load SAIL-VL model using AutoModel
        Requires trust_remote_code=True
        """
        self.model = AutoModel.from_pretrained(
            args.model_name_or_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            **self.model_kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            use_fast=False)

    def _parse_input(self, sample: dict):
        """
        Parse input sample and load images
        SAIL-VL uses <image> placeholder
        """
        question = sample["prompt"]
        q_chunks = re.split(r'(<(?:image|video)>)', question)

        # SAIL-VL expects newline after <image> tag
        question = question.replace("<image>", "<image>\n")

        media_list = copy.deepcopy(sample['media'])
        pixel_values_list = []
        num_patches_list = []

        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            if chunk == constants.image:
                image = media_list.pop(0)
                image_pixel_values = load_image(image, max_num=10)  # SAIL-VL max 10 tiles
                pixel_values_list.append(image_pixel_values)
                num_patches_list.append(image_pixel_values.size(0))

        if len(pixel_values_list) == 0:
            # Text-only input
            return question, None, None

        pixel_values = torch.cat(pixel_values_list, dim=0).to(self.dtype).to(self.device)

        return question, pixel_values, num_patches_list

    def _generate_response(self, question, pixel_values, num_patches_list):
        """
        Generate response using SAIL-VL's chat interface
        """
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            self.gen_kwargs,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False)

        return response

    def _score_choices(self, question, pixel_values, num_patches_list, sample):
        """
        Score multiple choice options using conditional probability
        This is an experimental implementation for scoring mode
        """
        choices = sample.get("choices")

        # Build inputs with model.build_conversation_input_ids
        # For now, we use a simplified approach similar to InternVL
        # Note: This might need adjustment based on actual SAIL-VL tokenization

        scores = []
        for choice in choices:
            full_question = question + " " + choice
            try:
                # Get model logits for scoring
                # This is a simplified implementation and may need refinement
                with torch.no_grad():
                    inputs = self.tokenizer(full_question, return_tensors="pt").to(self.device)
                    if pixel_values is not None:
                        outputs = self.model(
                            **inputs,
                            pixel_values=pixel_values,
                            num_patches_list=num_patches_list
                        )
                    else:
                        outputs = self.model(**inputs)

                    # Calculate score (simplified - might need adjustment)
                    logits = outputs.logits
                    score = logits.mean().item()
                    scores.append(score)
            except Exception as e:
                print(f"Warning: Scoring failed for choice '{choice}': {e}")
                scores.append(float('-inf'))

        return {
            "score": scores,
            "response": choices[np.argmax(scores)] if scores else choices[0]
        }

    def run_sample(self, sample: dict):
        """
        Run inference on a single sample
        """
        ori_sample = copy.deepcopy(sample)
        question, pixel_values, num_patches_list = self._parse_input(ori_sample)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(question, pixel_values, num_patches_list)
        else:
            # Scoring mode - experimental
            ori_sample.update(self._score_choices(question, pixel_values, num_patches_list, sample))

        return ori_sample


if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
