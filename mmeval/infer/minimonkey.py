import re
import copy
import torch

from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args

class TaskRunner(Task):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def load_model(self, args):
        self.model_name_or_path = f"mx262/{args.model_name_or_path}"
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True, use_fast=False)

    def run_sample(self, sample:dict):

        ori_sample = copy.deepcopy(sample)
        parsed_sample = self.parse_input(sample)

        if not self.args.score_target:
            ori_sample["response"] = self._generate_response(parsed_sample)
        else:
            raise NotImplementedError("Scoring is not supported")
        
        return ori_sample
    
    def _generate_response(self, data):
        generation_config = dict(do_sample=self.args.do_sample, max_new_tokens=self.args.max_new_tokens)
        response, history = self.model.chat(self.tokenizer, data["pixel_values"], data["target_aspect_ratio"], data["question"], generation_config, history=None, return_history=True)

        return response
    
    def parse_input(self, sample:dict):
        parsed_sample = copy.deepcopy(sample)

        question = sample["prompt"]
        # extract placeholder
        placeholders = re.findall(r'<[^>]*>', question)
        assert len(placeholders) == 1, f"VideoLLaMA2 supports one image or video, but got {len(placeholder)}"
        placeholder = placeholders[0]
        modality = "image" if placeholder == constants.image else "video"

        # remove the placeholder in the question
        question = question.replace(placeholder, "").strip()
        parsed_sample["question"] = question
        parsed_sample["modality"] = modality

        image = sample["media"][0]
        pixel_values, target_aspect_ratio = self.transform_image(image, min_num=4, max_num=12)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        parsed_sample["pixel_values"] = pixel_values
        parsed_sample["target_aspect_ratio"] = target_aspect_ratio

        return parsed_sample
    
    def transform_image(self, image_obj, input_size=448, min_num=1, max_num=12):
        transform = self.build_transform(input_size=input_size)
        images, target_aspect_ratio = self.dynamic_preprocess(image_obj, image_size=input_size, use_thumbnail=True, min_num=min_num, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values, target_aspect_ratio
        
    def build_transform(self, input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    def find_closest_aspect_ratio(self,aspect_ratio, target_ratios, width, height, image_size):
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
    
    def dynamic_preprocess(self,image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
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
        return processed_images, target_aspect_ratio

if __name__ == "__main__":
    args = parse_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()

        
