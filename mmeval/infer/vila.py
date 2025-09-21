import re
import copy
import torch
import numpy as np
import os
import sys
from typing import List, Optional, Union

# Add VILA path to sys.path for imports
vila_path = os.path.join(os.path.dirname(__file__), 'VILA')
if vila_path not in sys.path:
    sys.path.insert(0, vila_path)

from llava import conversation as clib
from llava.media import Image, Video
from llava.entry import load as load_vila_model
from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llava.conversation import SeparatorStyle

from mmeval.infer.task import Task
from mmeval.utils import constants
from mmeval.utils.argparser import parse_args, parse_model_kwargs, parse_gen_kwargs
from mmeval.utils.scorer import IncrementalLMScorer, target_tokens


def decode_time_token(text: str, *, duration: float, num_time_tokens: int, time_token_format: str) -> str:
    """Replace time tokens in text with actual timestamps."""
    for t in range(num_time_tokens):
        time_token = time_token_format.format(t=t)
        timestamp = round(t * duration / (num_time_tokens - 1), 2)
        text = text.replace(time_token, f"<{timestamp}>")

    # Handle out-of-range time tokens
    excess_pattern = re.compile(rf"<t(\d+)>")
    matches = excess_pattern.findall(text)
    for match in matches:
        t = int(match)
        if t >= num_time_tokens:
            timestamp = round(duration, 2)  # Map to the end of the video
            text = text.replace(f"<t{t}>", f"<{timestamp}>")

    return text


def configure_ps3_and_context_length(model):
    """Configure PS3 settings and adjust context length based on those settings."""
    # get PS3 configs from environment variables
    num_look_close = os.environ.get("NUM_LOOK_CLOSE", None)
    num_token_look_close = os.environ.get("NUM_TOKEN_LOOK_CLOSE", None)
    select_num_each_scale = os.environ.get("SELECT_NUM_EACH_SCALE", None)
    look_close_mode = os.environ.get("LOOK_CLOSE_MODE", None)
    smooth_selection_prob = os.environ.get("SMOOTH_SELECTION_PROB", None)

    # Set PS3 configs
    if num_look_close is not None:
        print("Num look close:", num_look_close)
        num_look_close = int(num_look_close)
        model.num_look_close = num_look_close
    if num_token_look_close is not None:
        print("Num token look close:", num_token_look_close)
        num_token_look_close = int(num_token_look_close)
        model.num_token_look_close = num_token_look_close
    if select_num_each_scale is not None:
        print("Select num each scale:", select_num_each_scale)
        select_num_each_scale = [int(x) for x in select_num_each_scale.split("+")]
        model.get_vision_tower().vision_tower.vision_model.max_select_num_each_scale = select_num_each_scale
    if look_close_mode is not None:
        print("Look close mode:", look_close_mode)
        model.look_close_mode = look_close_mode
    if smooth_selection_prob is not None:
        print("Smooth selection prob:", smooth_selection_prob)
        if smooth_selection_prob.lower() == "true":
            smooth_selection_prob = True
        elif smooth_selection_prob.lower() == "false":
            smooth_selection_prob = False
        else:
            raise ValueError(f"Invalid smooth selection prob: {smooth_selection_prob}")
        model.smooth_selection_prob = smooth_selection_prob

    # Adjust the max context length based on the PS3 config
    context_length = model.tokenizer.model_max_length
    if num_look_close is not None:
        context_length = max(context_length, num_look_close * 2560 // 4 + 1024)
    if num_token_look_close is not None:
        context_length = max(context_length, num_token_look_close // 4 + 1024)
    context_length = max(getattr(model.tokenizer, "model_max_length", context_length), context_length)
    model.config.model_max_length = context_length
    model.config.tokenizer_model_max_length = context_length
    model.llm.config.model_max_length = context_length
    model.llm.config.tokenizer_model_max_length = context_length
    model.tokenizer.model_max_length = context_length


class TaskRunner(Task):
    def __init__(self, args):
        self.args = args
        self.dtype = getattr(args, "dtype") or torch.bfloat16
        self.default_model_kwargs = {"device_map": "auto"}
        self.default_gen_kwargs = {}
        self.model_kwargs = parse_model_kwargs(args, self.default_model_kwargs)
        self.gen_kwargs = parse_gen_kwargs(args, self.default_gen_kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(args)
        
    def load_model(self, args):
        """Load VILA model using the llava.entry.load function"""
        # Extract model path and lora path from args if available
        model_path = args.model_name_or_path
        lora_path = getattr(args, 'lora_path', None)
        
        # Load the model
        if lora_path is None:
            self.model = load_vila_model(model_path, model_base=None)
        else:
            self.model = load_vila_model(lora_path, model_base=model_path)
        
        # Get tokenizer from model
        self.tokenizer = self.model.tokenizer
        
        # Override num_video_frames and video_max_tiles if specified
        num_video_frames = getattr(args, 'num_video_frames', -1)
        video_max_tiles = getattr(args, 'video_max_tiles', -1)
        
        if num_video_frames > 0:
            self.model.config.num_video_frames = num_video_frames
            
        if video_max_tiles > 0:
            self.model.config.video_max_tiles = video_max_tiles
            self.model.llm.config.video_max_tiles = video_max_tiles
        
        # Configure PS3 and adjust context length
        configure_ps3_and_context_length(self.model)
        
        # Set conversation mode
        conv_mode = getattr(args, 'conv_mode', 'auto')
        clib.default_conversation = clib.conv_templates[conv_mode].copy()

    def _parse_input(self, sample: dict):
        """Parse input sample to extract prompt and media"""
        prompt = sample["prompt"]
        media = copy.deepcopy(sample['media'])
        
        # Split prompt by media placeholders
        q_chunks = re.split(r'(<(?:image|video)>)', prompt)
        
        # Prepare conversation
        conv = clib.default_conversation.copy()
        user_role = conv.roles[0]
        
        processed_prompt = ""
        media_list = []
        
        for chunk in q_chunks:
            if len(chunk.strip()) == 0:
                continue
            
            if any(p in chunk for p in constants.all):
                # Handle media placeholder
                media_file = media.pop(0)
                
                if chunk == constants.image:
                    # Handle image
                    if isinstance(media_file, str):
                        media_obj = Image(media_file)
                    else:
                        media_obj = media_file
                    media_list.append(media_obj)
                    processed_prompt += chunk
                elif chunk == constants.video:
                    # Handle video
                    if isinstance(media_file, str):
                        media_obj = Video(media_file)
                    else:
                        media_obj = media_file
                    media_list.append(media_obj)
                    processed_prompt += chunk
                else:
                    raise ValueError(f"Unsupported media type: {chunk}")
            else:
                processed_prompt += chunk
        
        # Add user message to conversation
        conv.append_message(user_role, processed_prompt)
        
        # Get the final prompt
        if conv.sep_style == SeparatorStyle.LLAMA_3:
            assistant_role = conv.roles[1]
            conv.append_message(assistant_role, "")
        
        prompt_text = conv.get_prompt()
        
        return prompt_text, media_list

    def _generate_response(self, prompt_text: str, media_list: List[Union[Image, Video]]):
        """Generate response using VILA model"""
        # Prepare input for the model
        if media_list:
            # Multi-modal input
            prompt = media_list + [prompt_text]
        else:
            # Text-only input
            prompt = prompt_text
        
        # Generate response
        with torch.inference_mode():
            response = self.model.generate_content(prompt)
        
        # Check if we need to decode time tokens for video
        has_video = any(isinstance(media, Video) for media in media_list)
        if (has_video and 
            hasattr(self.model.config, 'num_time_tokens') and 
            self.model.config.num_time_tokens is not None and 
            hasattr(self.model.config, 'time_token_format') and 
            self.model.config.time_token_format is not None):
            
            # Get video duration (assuming first video)
            video_media = next((media for media in media_list if isinstance(media, Video)), None)
            if video_media:
                import cv2
                cap = cv2.VideoCapture(video_media.path)
                duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                # Decode time tokens
                response = decode_time_token(
                    response,
                    duration=duration,
                    num_time_tokens=self.model.config.num_time_tokens,
                    time_token_format=self.model.config.time_token_format,
                )
        
        return response
    
    def _score_choices(self, prompt_text: str, media_list: List[Union[Image, Video]], sample: dict):
        pass

    def run_sample(self, sample: dict):
        """Run inference on a single sample"""
        ori_sample = copy.deepcopy(sample)
        prompt_text, media_list = self._parse_input(ori_sample)
        
        if not self.args.score_target:
            # Generate response
            ori_sample["response"] = self._generate_response(prompt_text, media_list)
        else:
            # Score choices
            ori_sample.update(self._score_choices(prompt_text, media_list, sample))

        return ori_sample


def parse_vila_args():
    """Parse arguments for VILA inference with custom parameters"""
    import sys
    from mmeval.utils.argparser import parse_args as parse_base_args
    
    # Parse base arguments first
    args = parse_base_args()
    
    # Add custom VILA arguments by parsing sys.argv manually
    # This avoids conflicts with the existing argument parser
    conv_mode = "vicuna_v1"  # default
    lora_path = None
    num_video_frames = -1
    video_max_tiles = -1
    
    # Parse custom arguments from command line
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--conv_mode" and i + 1 < len(sys.argv):
            conv_mode = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--lora_path" and i + 1 < len(sys.argv):
            lora_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--num_video_frames" and i + 1 < len(sys.argv):
            num_video_frames = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--video_max_tiles" and i + 1 < len(sys.argv):
            video_max_tiles = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    # Add custom arguments to the base args
    args.conv_mode = conv_mode
    args.lora_path = lora_path
    args.num_video_frames = num_video_frames
    args.video_max_tiles = video_max_tiles
    
    return args

if __name__ == "__main__":
    args = parse_vila_args()
    model_evaluator = TaskRunner(args)
    model_evaluator.inference_dataset()
