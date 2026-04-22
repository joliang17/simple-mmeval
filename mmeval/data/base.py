import os
import re
import base64
import requests
from PIL import Image
from io import BytesIO
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple, Union
from jinja2 import Environment



class BaseDataset(ABC):
    """Dataset base class for loading and processing datasets.
    
    This class provides:
    1. Lazy data loading
    2. Memory-efficient processing
    3. Standard iteration helpers (__iter__, __getitem__, __len__)
    4. ID uniqueness enforcement
    5. Image resizing utilities
    6. Parallel processing support
    7. Circular data preparation placeholder
    """
    
    VIDEO_EXTENSIONS = {
        '.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv',
        '.mpeg', '.mpg', '.m4v', '.3gp', '.3g2', '.ts', '.mts', '.vob'
    }
    
    def __init__(self, args):
        """Initialize the dataset with lazy loading and parallel processing support."""
        self.parallel_per_task = args.parallel_per_task
        self.rank = args.rank
        self._shard_indices = []
        self._shard_length = 0
        
        # Load raw data and dataset's own template
        self._raw_dataset, self._dataset_template = self._load_raw_data(args)
        
        # Load user template if provided
        user_template_path = getattr(args, 'template', None)
        self._user_template = self._load_template(user_template_path) if user_template_path else None

    def setup_parallel(self, cache=None):
        if self._raw_dataset is None:
            self._shard_indices = []
            self._shard_length = 0
            return
        
        indices_to_run = list(range(len(self._raw_dataset)))
        if cache is not None:
            indices_to_run = [idx for idx in indices_to_run if idx not in cache.keys()]
            
        self._shard_indices = indices_to_run[self.rank::self.parallel_per_task]
        self._shard_length = len(self._shard_indices)    
    
    def _get_idx(self, index: int) -> int:
        assert index >= 0, "index must be non-negative"
        return self._shard_indices[index]

    def resize_image(
        self, 
        image: Image.Image, 
        size: Union[int, Tuple[int, int]], 
        padding: bool = True,
        fill_color: str = "black"
    ) -> Image.Image:
        """Resize a PIL Image object with optional padding.

        Parameters
        ----------
        image : Image.Image
            The PIL Image object to resize
        size : Union[int, Tuple[int, int]]
            Target size. If int, resize to (size, size). If tuple, resize to (width, height)
        padding : bool, default=True
            If True, maintain aspect ratio and pad to target size.
            If False, directly resize to target size.
        fill_color : str, default="black"
            Fill color for padding. Common colors: "black", "white", "red", "green", "blue", "gray"

        Returns
        -------
        Image.Image
            The resized PIL Image object
        """
        if isinstance(size, int):
            target_size = (size, size)
        else:
            target_size = size

        if not padding:
            return image.resize(target_size, Image.Resampling.LANCZOS)
        
        original_width, original_height = image.size
        target_width, target_height = target_size
        
        scale = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        padded_image = Image.new(image.mode, target_size, fill_color)
        
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2

        padded_image.paste(resized_image, (x_offset, y_offset))
        
        return padded_image

    def _load_template(self, template):
        """Load template from file path or use string directly."""
        if template is None:
            return None
        if os.path.exists(template):
            with open(template, "r") as file:
                return file.read()
        return template

    def _is_video(self, media_path: str) -> bool:
        """Check if the path is a video file by extension."""
        ext = os.path.splitext(media_path.split('?')[0])[-1].lower()
        return ext in self.VIDEO_EXTENSIONS

    def load_media(self, media) -> Union[Image.Image, str]:
        """Load media from path or return existing object. Returns PIL image for images, path for videos."""
        if isinstance(media, Image.Image):
            return media if media.mode == "RGB" else media.convert("RGB")
        
        if not isinstance(media, str):
            raise ValueError(f"Unsupported media type: {type(media)}")
        
        # Video: return path/URL directly
        if self._is_video(media):
            return media
        
        # Image: load with PIL
        if os.path.exists(media):
            return Image.open(media).convert("RGB")
        
        if media.startswith("http"):
            return Image.open(requests.get(media, stream=True).raw).convert("RGB")
        
        # Try base64 decode
        try:
            decoded = base64.b64decode(media)
            return Image.open(BytesIO(decoded)).convert("RGB")
        except Exception:
            raise ValueError(f"Unsupported media format: {media[:100]}...")
    
    def build_prompt(self, prompt_template: str, sample: Dict[str, Any]) -> str:
        """Build prompt from Jinja template and sample data.
        
        Parameters
        ----------
        prompt_template : str
            Jinja template string
        sample : Dict[str, Any]
            Sample variables for template rendering
            
        Returns
        -------
        str
            Rendered template string
        """
        env = Environment()
        env.globals.update({'zip': zip, 'enumerate': enumerate, 'len': len, 'range': range, 'list': list, 
        'dict': dict, 'str': str, 'int': int, 'float': float, 'bool': bool, 'sum': sum, 'max': max, 'min': min})
        template = env.from_string(prompt_template)
        return template.render(**sample)

    def _process_messages(self, message_list: List[Dict], media_list: List = None) -> List[Dict]:
        """Process all messages with sample-level media indexed by placeholder order."""
        # Prepare media paths with prefix
        media_list = media_list or []
        if getattr(self, 'media_dir', None) and media_list:
            media_list = [os.path.join(self.media_dir, media) if isinstance(media, str) else media for media in media_list]
        
        # Load default template (used as fallback)
        default_path = os.path.join(os.path.dirname(__file__), "default_template.txt")
        default_template = self._load_template(default_path)
        
        media_idx = 0
        processed_message_list = []
        
        for message in message_list:
            prompt = message.get("prompt")
            
            # Determine template and source (priority: user > dataset > default)
            if self._user_template:
                template, source = self._user_template, "User"
            elif prompt is None and self._dataset_template:
                template, source = self._dataset_template, "Dataset"
            elif prompt is None and default_template:
                template, source = default_template, "Default"
            elif prompt is None:
                raise ValueError("No prompt and template provided")
            else:
                template, source = None, None  # Use existing prompt
            
            # Build prompt from template if needed
            if template:
                try:
                    prompt = self.build_prompt(template, message)
                except Exception as e:
                    raise ValueError(f"{source} template rendering failed: {e}")
            
            # Validate and load media for this message's placeholders.
            placeholder_list = re.findall(r"<(video|image)>", prompt)
            remaining_media = len(media_list) - media_idx
            placeholder_count = len(placeholder_list)
            if placeholder_count > remaining_media:
                raise ValueError(
                    "Prompt/media mismatch while processing messages: "
                    f"message_index={len(processed_message_list)}, "
                    f"placeholders={placeholder_count}, remaining_media={remaining_media}"
                )
            processed_media_list = []
            for placeholder, media in zip(placeholder_list, media_list[media_idx:]):
                if placeholder == "image":
                    image = self.load_media(media)
                    if getattr(self, 'resize', None) is not None:
                        image = self.resize_image(image, self.resize)
                    processed_media_list.append(image)
                else:  # video
                    processed_media_list.append(media)
            media_idx += len(processed_media_list)
            
            message["prompt"] = prompt
            message["media"] = processed_media_list
            processed_message_list.append(message)
        
        if media_idx != len(media_list):
            raise ValueError(
                "Prompt/media mismatch while processing messages: "
                f"used_media={media_idx}, total_media={len(media_list)}"
            )
        
        return processed_message_list

    def convert_circular(self, **kwargs) -> Any:
        """Prepare dataset for circular evaluation.
        
        Parameters
        ----------
        **kwargs
            Additional arguments for circular data preparation configuration
            
        Returns
        -------
        Any
            Results from circular data preparation (implementation dependent)
        """

        raise NotImplementedError("convert_circular not implemented.")

    @abstractmethod
    def _load_raw_data(self, args) -> tuple:
        """Load raw data and dataset-specific template.
        
        Parameters
        ----------
        args
            arguments specific to the dataset type
            
        Returns
        -------
        tuple[Any, str | None]
            (dataset, dataset_template) where:
            - dataset: object that supports indexing (dataset[i]) and len()
            - dataset_template: dataset's own template (e.g., HF jinja_template) or None
        """

    @abstractmethod
    def _process_sample(self, idx: int) -> Dict[str, Any]:
        """Process sample by index - to be implemented by subclasses.
        
        Subclasses should:
        1. Get raw item: raw_item = self._raw_dataset[index]
        2. Process the raw item into common schema
        
        Parameters
        ----------
        idx : int
            Global sample index (not local to this worker)
            
        Returns
        -------
        Dict[str, Any]
            Processed data item following the common schema
        """

    # Iteration helpers with lazy loading and parallel processing
    def __iter__(self):
        """Iterate over dataset samples assigned to this worker with lazy loading."""
        for i in range(self._shard_length):
            idx = self._get_idx(i)
            sample = self._process_sample(idx)
            # TODO: add more checks later (mandatory fields)
            assert "eval-id" in sample, "eval-id is mandatory."
            assert "messages" in sample, "messages is mandatory."
            yield sample

    def __getitem__(self, index):
        """Get sample by local index with lazy loading.
        
        Parameters
        ----------
        index : int
            Local index within this worker's data subset
            
        Returns
        -------
        Dict[str, Any]
            Processed sample
        """
        sample = self._process_sample(self._get_idx(index))
        # TODO: add more checks later (mandatory fields)
        assert "eval-id" in sample, "eval-id is mandatory"
        assert "messages" in sample, "messages is mandatory."
        return sample

    def __len__(self):
        """Get number of samples assigned to this worker."""
        return self._shard_length
    
    @property
    def global_length(self):
        """Get total number of samples in the original dataset."""
        return len(self._raw_dataset) if self._raw_dataset else 0
