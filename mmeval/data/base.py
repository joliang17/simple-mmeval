import os
import base64
import requests
from PIL import Image
from io import BytesIO
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple, Union



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
    
    def __init__(self, args):
        """Initialize the dataset with lazy loading and parallel processing support.
        
        Parameters
        ----------

        """
        self.parallel_per_task = args.parallel_per_task
        self.rank = args.rank
        
        self._raw_dataset = self._load_raw_data(args)
        
        # self._setup_parallel()

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

    def load_image(self, f) -> Image.Image:
        """Load image from path with PIL."""
        if isinstance(f, Image.Image):
            return f if f.mode == "RGB" else f.convert("RGB")
        
        if isinstance(f, str) and os.path.exists(f):
            return Image.open(f).convert("RGB")
        
        if isinstance(f, str) and f.startswith("http"):
            return Image.open(requests.get(f, stream=True).raw).convert("RGB")
        
        if isinstance(f, str):
            try:
                decoded = base64.b64decode(f)
                return Image.open(BytesIO(decoded)).convert("RGB")
            except Exception:
                pass
        
        raise NotImplementedError(f"Unsupported image format: {f}")
    
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
    def _load_raw_data(self, args) -> Any:
        """Load raw data from the data source.
        
        Returns a dataset object that supports indexing and length,
        but doesn't necessarily load all data into memory at once.
        
        Parameters
        ----------
        args
            arguments specific to the dataset type
            
        Returns
        -------
        Any
            Dataset object that supports indexing (dataset[i]) and len()
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
            assert "prompt" in sample, "prompt is mandatory."
            assert "media" in sample, "media is mandatory."
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
        assert "prompt" in sample, "prompt is mandatory."
        assert "media" in sample, "media is mandatory."
        return sample

    def __len__(self):
        """Get number of samples assigned to this worker."""
        return self._shard_length
    
    @property
    def global_length(self):
        """Get total number of samples in the original dataset."""
        return len(self._raw_dataset) if self._raw_dataset else 0