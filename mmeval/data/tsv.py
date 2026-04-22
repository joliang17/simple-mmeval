import os
import re
import ast
import string
import pandas as pd
from typing import Dict, Any
from mmeval.data.base import BaseDataset
from mmeval.data.utils import download_tsv


IMG_PLACEHOLDER_RE = re.compile(
    r"""
        <img[^>]*>            |  # any <img …>
        <image[^>]*>          |  # catch-all <image …>  (covers <image 1>, <image_2>, …)
        <imagehere>           |  # <ImageHere>
        <img_plh>             |  # <IMG_PLH>
        <img_context>            # <IMG_CONTEXT>

    """,
    re.IGNORECASE | re.VERBOSE,
)
MEDIA_PLACEHOLDER_RE = re.compile(r"<(?:video|image)>")

class TSVDataset(BaseDataset):
    """Dataset class for loading TSV files.
    
    This class provides a bridge between TSV files and the mmeval Dataset interface.
    """
    
    def __init__(self, args):
        """Initialize the TSV dataset with parallel processing support.
        
        Parameters
        ----------
        args: argparse.Namespace
            Arguments from argparse containing dataset configuration
        """
        self.dataset_dir = os.getenv('DATASET_DIR') or "./datasets"
        self.dataset_url = None
        self.resize = args.resize
        
        if args.dataset.startswith("http"):
            self.file_name = args.dataset.split('/')[-1].replace('.tsv', '')
            self.dataset_url = args.dataset
        elif os.path.exists(args.dataset):
            self.file_name = args.dataset.split('/')[-1].replace('.tsv', '')
        else:
            self.file_name = args.dataset

        super().__init__(args)
    
    def _load_raw_data(self, args) -> tuple:
        """Load raw data from TSV files.
        
        Returns
        -------
        tuple[pd.DataFrame, None]
            (dataset, None) - TSV has no dataset-specific template
        """
        data_file = os.path.join(self.dataset_dir, f"{self.file_name}.tsv")

        if not os.path.exists(data_file) and self.dataset_url:
            download_tsv(self.dataset_url, data_file)
            
        dataset = pd.read_csv(data_file, sep='\t')

        if "eval-id" not in dataset.columns:
            dataset["eval-id"] = range(len(dataset))
        
        return dataset, None  # No dataset-specific template

    def _extract_media_paths(self, sample: Dict[str, Any]) -> list:
        """Extract media paths/data from sample without loading.
        
        Parameters
        ----------
        sample : Dict[str, Any]
            Sample dictionary
            
        Returns
        -------
        list
            List of media paths or base64 strings
        """
        media_paths = []
        
        # Priority 1: Check for image_url
        if 'image_url' in sample and pd.notna(sample['image_url']):
            image_url = sample['image_url']
            if image_url.startswith('[') and image_url.endswith(']'):
                media_paths = ast.literal_eval(image_url)
            else:
                media_paths = [image_url]
                        
        # Priority 2: Check for base64 image data
        elif 'image' in sample and pd.notna(sample['image']):
            image = sample['image']
            if image.startswith('[') and image.endswith(']'):
                media_paths = ast.literal_eval(image)
            else:
                media_paths = [image]
        
        return media_paths

    def _process_sample(self, index: int) -> Dict[str, Any]:
        """Process a raw TSV sample to match unified format.
        
        Parameters
        ----------
        index : int
            Global index of the sample in the dataset
            
        Returns
        -------
        Dict[str, Any]
            Processed sample with messages list
        """
        sample = self._raw_dataset.iloc[index].to_dict()
        # Replace NaN values with None for JSON serialization
        sample = {k: (None if pd.isna(v) else v) for k, v in sample.items()}
        media_list = self._extract_media_paths(sample)
        question = str(sample['question'])

        # Normalize image placeholders
        if IMG_PLACEHOLDER_RE.search(question):
            question = IMG_PLACEHOLDER_RE.sub("<image>", question)
        elif media_list:
            question = f'{"<image>" * len(media_list)} {question}'.strip()
        
        placeholder_count = len(MEDIA_PLACEHOLDER_RE.findall(question))
        media_count = len(media_list)
        if placeholder_count > media_count:
            # If prompt has more placeholders than media, keep only media_count
            # placeholders and move them to the prompt prefix.
            question_without_placeholders = MEDIA_PLACEHOLDER_RE.sub(" ", question)
            question_without_placeholders = re.sub(r"\s+", " ", question_without_placeholders).strip()
            placeholder_prefix = "<image>" * media_count
            question = f"{placeholder_prefix} {question_without_placeholders}".strip() if question_without_placeholders else placeholder_prefix
            placeholder_count = media_count

        if placeholder_count < media_count:
            eval_id = sample.get("eval-id", index)
            question_preview = question.replace("\n", " ")[:200]
            raise ValueError(
                "Prompt/media mismatch for eval-id "
                f"{eval_id}: placeholders={placeholder_count}, media={media_count}, "
                f"question='{question_preview}'"
            )

        # Build options dict and choices list
        options = {
            choice_index: sample[choice_index] for choice_index in string.ascii_uppercase
            if choice_index in sample and not pd.isna(sample[choice_index])
        }

        # Build message dict for template rendering
        message = {"question": question}
        if options:
            message["options"] = options
            message["choices"] = list(options.keys())
        
        hint = sample.get("hint", None)
        if hint and pd.notna(hint):
            message["hint"] = hint

        # Process message using base class method with sample-level media
        sample["messages"] = self._process_messages([message], media_list)
        
        # Clean up original fields
        sample.pop("image", None)
        sample.pop("image_url", None)
        sample["media"] = media_list
        
        return sample

    def __repr__(self):
        if self.parallel_per_task > 1:
            return f"{self.file_name}(rank={self.rank}/{self.parallel_per_task}, local={len(self)}, global={self.global_length})"
        else:
            return f"{self.file_name}(samples={len(self)})"
    
    def __str__(self):
        return self.__repr__()